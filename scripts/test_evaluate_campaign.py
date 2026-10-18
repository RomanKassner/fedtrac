"""evaluate_campaign.py: the published graph as the model, the two baselines in the clear, the means by size, a result belongs to one run."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_campaign import (COLUMNS, INSTRUMENT, campaign_runs, envelope, evaluate_folder,  # noqa: E402
                               graph_from_traces, is_current, random_partition, result_path, score_clear,
                               spread, summarize, table_rows, to_event_log, token_generalization)


def net_of(dfg, start, end):
    """pm4py's Petri net of a graph, the model the instrument scores."""
    from pm4py.objects.conversion.dfg import converter as dfg_to_net
    return dfg_to_net.apply(dfg, parameters={"start_activities": start, "end_activities": end})


def cluster(size, f=None, rel=None, **more):
    c = {"size": size, **more}
    if f is not None:
        c.update(fitness=1.0, precision=f, f_score=f, simplicity=0.5)
    if rel is not None:
        c.update(rel=rel, rho=1.0)
    return c


def test_a_tier_is_the_mean_over_its_clusters_weighted_by_size():
    s = summarize([cluster(300, f=0.9, rel=2.0, pruned_activities=1, edges=10), cluster(100, f=0.5, rel=6.0, pruned_activities=2, edges=30)])
    assert (s["clusters"], s["cases"], s["models_scored"], s["pruned_activities"]) == (2, 400, 2, 3)
    assert s["f_score"] == 0.8 and s["precision"] == 0.8 and s["fitness"] == 1.0 and s["rel"] == 3.0 and s["edges"] == 15.0


def test_an_empty_cluster_is_no_cluster_and_an_unscored_model_leaves_the_model_means_but_not_the_graph_means():
    s = summarize([cluster(0), cluster(300, f=0.9, rel=2.0, edges=10),
                   {"size": 100, "rel": 6.0, "rho": 0.5, "edges": 30, "no_score": "no alignment within the limit"}])
    assert (s["clusters"], s["cases"], s["models_scored"]) == (2, 400, 1)
    assert s["f_score"] == 0.9          # the scored cluster alone
    assert s["rel"] == 3.0 and s["edges"] == 15.0   # the graph needs no model, so both clusters count
    assert summarize([cluster(0)])["f_score"] is None


def test_the_graph_of_the_traces_carries_the_protocol_s_edge_filter():
    # count 2: a pair seen once leaves; one fiftieth: a pair seen twice leaves when the largest count is 101
    dfg, start, end = graph_from_traces([["a", "b", "c"]] * 101 + [["a", "e", "c"]] * 2 + [["a", "d", "c"]])
    assert dfg == {("a", "b"): 101.0, ("b", "c"): 101.0}
    assert start == {"a": 104.0} and end == {"c": 104.0}      # start and end counts are never filtered
    env = envelope(dfg, start, end)
    assert env["edges"][0] == {"from": ">", "to": "a", "weight": 104.0} and env["edges"][-1] == {"from": "c", "to": "|", "weight": 104.0}


def test_a_random_partition_keeps_the_sizes_and_every_trace_and_follows_its_seed():
    traces = [[str(i)] for i in range(10)]
    parts = random_partition(traces, [3, 7], 1)
    assert [len(p) for p in parts] == [3, 7]
    assert sorted(t[0] for p in parts for t in p) == [str(i) for i in range(10)]
    assert random_partition(traces, [3, 7], 1) == parts
    assert random_partition(traces, [3, 7], 2) != parts


def test_the_graph_as_the_model_scores_as_the_count_by_hand():
    # The graph a->b, b->a, a->c, a the start and c the end activity. Precision by hand over the empty and every
    # proper prefix, weighted: at = 4*1 + 4*2 + 2*1 + 2*2 = 18, escaping = 2, so 1 - 2/18 = 0.8889. Relevance by
    # hand: a,b,a,c costs log2(3) + log2(3/2), a,c costs log2(3/2).
    traces = [["a", "b", "a", "c"]] * 2 + [["a", "c"]] * 2
    r = score_clear(traces, limit_s=60)
    assert (r["size"], r["edges"], r["activities"], r["pruned_activities"]) == (4, 3, 3, 0)
    assert r["fitness"] == 1.0 and r["precision"] == 0.8889 and r["f_score"] == 0.9412
    assert r["rel"] == 1.3774 and r["rho"] == 1.0
    assert 0 < r["simplicity"] <= 1


def test_an_activity_on_no_start_to_end_path_leaves_the_model_and_its_events_count_against_the_fitness():
    # a->b is seen three times and stays; each pair after b is seen once and leaves, so b reaches no end activity,
    # and d and e are end activities that no start activity reaches: the pruning removes the three, the model is
    # a->c alone, and the three traces through b do not fit it entirely
    traces = [["a", "c"]] * 100 + [["a", "b", "c"], ["a", "b", "d"], ["a", "b", "e"]]
    r = score_clear(traces, limit_s=60)
    assert r["pruned_activities"] == 3 and r["edges"] == 1 and r["activities"] == 2
    assert r["fitness"] < 1.0 and r["rho"] == round(100 / 103, 4)


def test_the_relevance_reads_the_pruned_graph_the_graph_every_other_measure_reads():
    # The graph before the pruning holds b, d and e as well, so its uniform background model codes a trace it
    # does not allow over five activities instead of two.
    from bridge.metrics import entropic_relevance
    from bridge.utils import prune_to_start_end_paths
    traces = [["a", "c"]] * 100 + [["a", "b", "c"], ["a", "b", "d"], ["a", "b", "e"]]
    dfg, start, end = graph_from_traces(traces)
    pruned = prune_to_start_end_paths(dfg, start, end)[:3]
    on_pruned = round(entropic_relevance(traces, envelope(*pruned))["rel"], 4)
    on_opened = round(entropic_relevance(traces, envelope(dfg, start, end))["rel"], 4)
    assert on_pruned != on_opened
    assert score_clear(traces, limit_s=60)["rel"] == on_pruned


def tier_folder(tmp_path):
    folder = tmp_path / "federated"
    folder.mkdir()
    for cid, traces in ((0, [["a", "b", "c"]] * 4), (1, [["a", "c", "b"]] * 4), (2, [])):
        dfg, start, end = graph_from_traces(traces)
        (folder / f"cluster_{cid}_dfg.json").write_text(json.dumps(envelope(dfg, start, end)))
        (folder / f"cluster_{cid}_traces.json").write_text(json.dumps({"traces": traces}))
    return folder


def test_a_folder_result_carries_the_clusters_the_whole_log_and_five_random_partitions_of_the_same_sizes(tmp_path):
    d = evaluate_folder(tier_folder(tmp_path), limit_s=60)
    assert d["instrument"] == INSTRUMENT
    assert [c["cluster_id"] for c in d["clusters"]] == [0, 1, 2] and d["clusters"][2] == {"size": 0, "cluster_id": 2}
    assert d["summary"]["clusters"] == 2 and d["summary"]["cases"] == 8 and d["summary"]["f_score"] == 1.0
    assert d["whole"]["size"] == 8 and d["whole"]["fitness"] == 1.0 and d["whole"]["edges"] == 4
    assert [r["seed"] for r in d["random"]["seeds"]] == [1, 2, 3, 4, 5]
    assert all((r["clusters"], r["cases"]) == (2, 8) for r in d["random"]["seeds"])
    assert d["random"]["min"]["precision"] <= d["random"]["mean"]["precision"] <= d["random"]["max"]["precision"]


def test_the_spread_over_the_partitions_is_mean_lowest_and_highest():
    s = spread([{"precision": 0.5, "edges": 10}, {"precision": 0.7, "edges": 20}, {"precision": None, "edges": 30}], keys=("precision", "edges"))
    assert s["mean"] == {"precision": 0.6, "edges": 20.0} and s["min"] == {"precision": 0.5, "edges": 10} and s["max"] == {"precision": 0.7, "edges": 30}


def campaign(tmp_path):
    (tmp_path / "summary.tsv").write_text("cell\texit\nms_a_rp_bigram_s42\t0\nms_a_rp_ef_s42\t1\nms_a_rp_bigram_s7\t0\n")
    (tmp_path / "manifest.tsv").write_text(
        "cell\trun_dir\n"
        "ms_a_rp_bigram_s42\t-\n"
        "ms_a_rp_bigram_s42\toutput/runs/uni_ms_a_rp_bigram_s42_2026-10-04_20-00-00\n"
        "ms_a_rp_bigram_s42\toutput/runs/uni_ms_a_rp_bigram_s42_2026-10-06_01-00-00\n"
        "ms_a_rp_ef_s42\toutput/runs/uni_ms_a_rp_ef_s42_2026-10-04_21-00-00\n"
        "ms_a_rp_bigram_s7\toutput/runs/uni_ms_a_rp_bigram_s7_2026-10-04_22-00-00\n")
    (tmp_path / "model_quality").mkdir()
    return tmp_path


def test_the_runs_are_the_last_run_directory_of_every_cell_that_exited_0(tmp_path):
    runs = campaign_runs(campaign(tmp_path))
    assert runs == {"ms_a_rp_bigram_s42": "output/runs/uni_ms_a_rp_bigram_s42_2026-10-06_01-00-00",
                    "ms_a_rp_bigram_s7": "output/runs/uni_ms_a_rp_bigram_s7_2026-10-04_22-00-00"}


def test_a_result_of_an_earlier_run_or_of_the_earlier_instrument_is_not_current_and_stays_out_of_the_table(tmp_path):
    c = campaign(tmp_path)
    runs = campaign_runs(c)
    summary = summarize([cluster(10, f=0.8, rel=1.0, edges=5)])
    rnd = {"seeds": [], "mean": {"precision": 0.6}, "min": {"precision": 0.5}, "max": {"precision": 0.7}}
    old_run = {"instrument": INSTRUMENT, "run_dir": "uni_ms_a_rp_bigram_s42_2026-10-04_20-00-00", "summary": summary,
               "whole": {"f_score": 0.7, "rel": 2.0}, "random": rnd}
    path = result_path(c, "ms_a_rp_bigram_s42", "federated")
    path.write_text(json.dumps(old_run))
    assert not is_current(path, runs["ms_a_rp_bigram_s42"])
    old_instrument = dict(old_run, run_dir="uni_ms_a_rp_bigram_s42_2026-10-06_01-00-00", global_={}, instrument=None)
    path.write_text(json.dumps(old_instrument))
    assert not is_current(path, runs["ms_a_rp_bigram_s42"]) and table_rows(c, runs) == []
    path.write_text(json.dumps(dict(old_run, run_dir="uni_ms_a_rp_bigram_s42_2026-10-06_01-00-00")))
    rows = table_rows(c, runs)
    assert len(rows) == 1 and list(rows[0]) == COLUMNS
    assert (rows[0]["cell"], rows[0]["tier"], rows[0]["f_score"], rows[0]["edges"], rows[0]["whole_f_score"], rows[0]["whole_rel"],
            rows[0]["random_precision_mean"], rows[0]["random_precision_max"]) == (
        "ms_a_rp_bigram_s42", "federated", 0.8, 5.0, 0.7, 2.0, 0.6, 0.7)


# --- Buijs's token-based generalization ----

def test_the_generalization_is_buijs_s_token_based_measure_as_pm4py_computes_it():
    # The net of a->b (4) and a->c (9), a the start and b, c the end activities, holds five transitions; the
    # start transition fires 13 times, a->b and b's silent transition 4 times, a->c and c's 9 times, so the value
    # is 1 - (1/sqrt(13) + 1/2 + 1/2 + 1/3 + 1/3) / 5 = 0.6111966.
    net, im, fm = net_of({("a", "b"): 4.0, ("a", "c"): 9.0}, {"a": 13.0}, {"b": 4.0, "c": 9.0})
    log = to_event_log([["a", "b"]] * 4 + [["a", "c"]] * 9)
    assert token_generalization(net, im, fm, log) == pytest.approx(1 - (13 ** -0.5 + 0.5 + 0.5 + 1 / 3 + 1 / 3) / 5)
    assert round(token_generalization(net, im, fm, log), 4) == 0.6112
    # a transition that never fires contributes 1: the same net against the four traces a,b alone leaves c and its
    # silent transition unfired, so 1 - (1/2 + 1/2 + 1/2 + 1 + 1) / 5 = 1 - 3.5/5 = 0.3
    assert token_generalization(net, im, fm, to_event_log([["a", "b"]] * 4)) == pytest.approx(0.3)


def test_the_generalization_of_a_cluster_is_the_one_of_the_net_the_other_three_measures_are_scored_on():
    r = score_clear([["a", "b", "a", "c"]] * 2 + [["a", "c"]] * 2, limit_s=60)
    # the net of a->b, b->a, a->c (a the start, c the end activity) holds five transitions and each one fires: the
    # start transition of a 4 times (once per trace), a->b twice, b->a twice, a->c 4 times, the silent transition
    # of c 4 times, so 1 - (1/2 + 1/sqrt(2) + 1/sqrt(2) + 1/2 + 1/2) / 5 = 1 - 2.9142136/5 = 0.4171573
    assert r["generalization"] == 0.4172
    assert r["fitness"] == 1.0 and r["precision"] == 0.8889        # unchanged by the fourth measure
    assert summarize([dict(r, size=4)])["generalization"] == 0.4172


def test_the_csv_appends_the_generalization_and_keeps_the_order_of_every_earlier_column():
    old = ["cell", "tier", "run_dir", "clusters", "cases", "models_scored", "pruned_activities",
           "fitness", "precision", "f_score", "simplicity", "edges", "activities", "rel", "rho",
           "whole_fitness", "whole_precision", "whole_f_score", "whole_simplicity", "whole_edges",
           "whole_activities", "whole_rel", "whole_rho", "whole_pruned_activities",
           *(f"random_{k}_{s}" for k in ("fitness", "precision", "f_score", "simplicity", "edges", "rel")
             for s in ("mean", "min", "max"))]
    assert len(old) == 42 and COLUMNS[:42] == old                 # every reader of model_quality.csv keeps working
    assert COLUMNS[42:] == ["generalization", "whole_generalization", "random_generalization_mean",
                            "random_generalization_min", "random_generalization_max"]


def test_the_table_carries_the_generalization_of_the_summary_the_whole_log_and_the_partitions(tmp_path):
    c = campaign(tmp_path)
    runs = campaign_runs(c)
    summary = dict(summarize([cluster(10, f=0.8, rel=1.0, edges=5, generalization=0.42)]))
    rnd = {"seeds": [], "mean": {"generalization": 0.3}, "min": {"generalization": 0.2}, "max": {"generalization": 0.4}}
    result_path(c, "ms_a_rp_bigram_s42", "federated").write_text(json.dumps(
        {"instrument": INSTRUMENT, "run_dir": "uni_ms_a_rp_bigram_s42_2026-10-06_01-00-00", "summary": summary,
         "whole": {"f_score": 0.7, "generalization": 0.55}, "random": rnd}))
    rows = table_rows(c, runs)
    assert len(rows) == 1 and list(rows[0]) == COLUMNS
    assert (rows[0]["generalization"], rows[0]["whole_generalization"], rows[0]["random_generalization_mean"],
            rows[0]["random_generalization_min"], rows[0]["random_generalization_max"]) == (0.42, 0.55, 0.3, 0.2, 0.4)
