import numpy as np
import pytest

from midas_hand_api.tactile import paxini


class _FakeSerial:
    def __init__(self, response: bytes) -> None:
        self.response = bytearray(response)
        self.written = bytearray()
        self.port = "fake"

    @property
    def in_waiting(self) -> int:
        return len(self.response)

    def read(self, length: int) -> bytes:
        data = bytes(self.response[:length])
        del self.response[:length]
        return data

    def write(self, data: bytes) -> int:
        self.written.extend(data)
        return len(data)

    def flush(self) -> None:
        pass


def _auto_push_frame(payload: bytes, *, status: int = 0) -> bytes:
    body = bytes([status]) + payload
    frame = b"\xAA\x56\x00" + len(body).to_bytes(2, "little") + body
    return frame + bytes([paxini._lrc(frame)])


def _register_response(address: int, data: bytes) -> bytes:
    frame = (
        b"\xAA\x55\x00\x03"
        + address.to_bytes(2, "little")
        + len(data).to_bytes(2, "little")
        + data
    )
    return frame + bytes([paxini._lrc(frame)])


def test_parse_auto_push_status_and_length_offsets() -> None:
    payload = bytes(range(12))
    frame = _auto_push_frame(payload)

    assert int.from_bytes(frame[3:5], "little") == len(payload) + 1
    assert frame[5] == 0
    assert paxini._parse_frame_payload(frame) == payload


def test_nonzero_auto_push_status_is_board_error_not_payload() -> None:
    frame = _auto_push_frame(b"force bytes", status=0x04)

    with pytest.raises(paxini._AutoPushBoardError) as caught:
        paxini._parse_frame_payload(frame)

    assert caught.value.status == 0x04
    assert caught.value.frame_length == len(frame)


def test_parse_auto_push_rejects_declared_length_mismatch() -> None:
    frame = _auto_push_frame(b"abc") + b"extra"

    with pytest.raises(ValueError, match="length mismatch"):
        paxini._parse_frame_payload(frame)


def test_active_module_bitmap_uses_dense_28_module_mapping() -> None:
    assert paxini._active_module_indices(bytes([0b01000100, 0b01000100, 0, 0])) == [
        2,
        6,
        10,
        14,
    ]


def test_read_adapter_register_uses_aa55_response_layout() -> None:
    serial = _FakeSerial(_register_response(0x0030, b"\x7f\x00\x34\x00"))

    data = paxini._read_adapter_register(
        serial, 0x0030, 4, timeout_s=0.1
    )

    assert data == b"\x7f\x00\x34\x00"
    assert bytes(serial.written) == paxini._build_adapter_read_frame(0x0030, 4)


def test_parse_digits_uses_discovered_variable_point_counts() -> None:
    fingers = [
        paxini._Finger("thumb", 3, 2),
        paxini._Finger("index", 7, 1),
    ]
    thumb_points = bytes([1, 2, 3, 0xFF, 0xFE, 4])
    index_points = bytes([5, 6, 7])
    payload = bytes(6) + thumb_points + bytes(6) + index_points

    parsed = paxini._parse_digits(
        payload, fingers, scale_n=0.1, signed_z=False
    )

    np.testing.assert_allclose(
        parsed["thumb"],
        np.array([[0.1, 0.2, 0.3], [-0.1, -0.2, 0.4]]),
    )
    np.testing.assert_allclose(
        parsed["index"], np.array([[0.5, 0.6, 0.7]])
    )


def test_parse_digits_requires_exact_discovered_payload_size() -> None:
    fingers = [paxini._Finger("index", 7, 1)]

    with pytest.raises(ValueError, match="requires exactly 9"):
        paxini._parse_digits(
            bytes(10), fingers, scale_n=0.1, signed_z=False
        )


def test_discover_fingers_reads_active_slots_and_point_counts(monkeypatch) -> None:
    counts = bytearray(56)
    counts[2 * 2 : 2 * 2 + 2] = (127).to_bytes(2, "little")
    counts[6 * 2 : 6 * 2 + 2] = (52).to_bytes(2, "little")
    responses = {
        paxini._ADDR_ACTIVE_MODULES: bytes([0b01000100, 0, 0, 0]),
        paxini._ADDR_DISTRIBUTION_POINT_COUNTS: bytes(counts),
        paxini._ADDR_DATA_TYPE: b"\x03",
    }

    def fake_read(_serial, address, length, *, timeout_s):
        assert timeout_s == 0.25
        return responses[address][:length]

    monkeypatch.setattr(paxini, "_read_adapter_register", fake_read)
    configured = paxini._build_fingers(["thumb", "index"])

    discovered = paxini._discover_fingers(
        object(), configured, timeout_s=0.25
    )

    assert [
        (finger.name, finger.device_address, finger.force_points)
        for finger in discovered
    ] == [("thumb", 3, 127), ("index", 7, 52)]
