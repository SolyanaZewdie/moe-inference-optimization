from src.dispatch.routing_traces import (
    create_routing_trace,
    save_routing_trace,
    load_routing_trace
)


def test_create_routing_trace():
    decisions = [
        [[0, 2], [1, 2]],
        [[1, 3], [0, 3]]
    ]

    trace = create_routing_trace(decisions)

    expected = [
        "layer_0_expert_0",
        "layer_0_expert_1",
        "layer_0_expert_2",
        "layer_1_expert_0",
        "layer_1_expert_1",
        "layer_1_expert_3"
    ]

    assert trace == expected


def test_experts_from_different_layers_are_distinct():
    decisions = [
        [[0]],
        [[0]]
    ]

    trace = create_routing_trace(decisions)

    assert trace[0] != trace[1]


def test_save_and_load_routing_trace(tmp_path):
    trace = [
        "layer_0_expert_1",
        "layer_0_expert_3",
        "layer_1_expert_2"
    ]

    file_path = tmp_path / "routing_trace.json"

    save_routing_trace(trace, file_path)
    loaded_trace = load_routing_trace(file_path)

    assert loaded_trace == trace


def test_empty_routing_trace():
    trace = create_routing_trace([])

    assert trace == []


def test_routing_trace_removes_duplicate_expert_requests():
    decisions = [
        [[2, 0], [1, 2], [0, 1]]
    ]

    trace = create_routing_trace(decisions)

    assert trace == [
        "layer_0_expert_0",
        "layer_0_expert_1",
        "layer_0_expert_2"
    ]


def test_routing_trace_uses_ascending_expert_order():
    decisions = [
        [[3, 1], [2, 3], [0, 1]]
    ]

    trace = create_routing_trace(decisions)

    assert trace == [
        "layer_0_expert_0",
        "layer_0_expert_1",
        "layer_0_expert_2",
        "layer_0_expert_3"
    ]