from collections.abc import Sequence

from msgspec.json import decode as json_decode
from msgspec.json import encode as json_encode
from msgspec.msgpack import decode as msgpack_decode
from msgspec.msgpack import encode as msgpack_encode

from vicinity.datatypes import ItemType


def encode_msgspec(data: Sequence[str], itemtype: ItemType) -> bytes:
    """Encode a list of strings to a format supported by msgpack."""
    if itemtype == ItemType.JSON:
        return json_encode(data)
    elif itemtype == ItemType.MSGPACK:
        return msgpack_encode(data)
    else:
        raise ValueError(f"Unknown item type: {itemtype}")


def decode_msgpack(data: bytes, itemtype: ItemType) -> Sequence[str]:
    """Decode bytes to a list of strings based on the item type."""
    if itemtype == ItemType.JSON:
        return json_decode(data)
    elif itemtype == ItemType.MSGPACK:
        return msgpack_decode(data)
    else:
        raise ValueError(f"Unknown item type: {itemtype}")
