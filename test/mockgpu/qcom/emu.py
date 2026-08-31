import ctypes
from collections.abc import Iterator
from dataclasses import dataclass
from tinygrad.runtime.autogen import libc, mesa

@dataclass(frozen=True)
class IR3Register:
  kind: str
  number: int     # GPR
  component: int  # SWIZ

@dataclass(frozen=True)
class IR3Source:
  value: IR3Register | int | float
  repeat: bool = False
  absolute: bool = False
  negate: bool = False
  last: bool = False

@dataclass(frozen=True)
class IR3Instruction:
  name: str
  dst: IR3Register | None
  srcs: tuple[IR3Source, ...]
  sy: bool = False
  ss: bool = False
  jp: bool = False
  sat: bool = False
  repeat: int = 0
  ul: bool = False
  ei: bool = False

def _read_field(fields: Iterator[tuple[str, str | int]], expected_name: str) -> str | int:
  try: name, value = next(fields)
  except StopIteration: raise ValueError(f'missing IR3 field {expected_name}') from None
  if name != expected_name: raise ValueError(f'expected IR3 field {expected_name}, got {name}')
  return value

def _decode_gpr(value: int, half: bool = False) -> IR3Register:
  if not 0 <= value < 256: raise ValueError(f'invalid GPR encoding {value}')
  number, component = value >> 2, value & 0x3
  if number in (61, 62): raise NotImplementedError(f'unsupported special register encoding {number}')
  return IR3Register('hr' if half else 'r', number, component)

def _read_gpr(fields: Iterator[tuple[str, str | int]], field: str, half: bool = False) -> IR3Register:
  encoded = int(_read_field(fields, field))
  register = _decode_gpr(encoded, half)
  if int(_read_field(fields, 'GPR')) != register.number: raise ValueError(f'invalid {field} GPR')
  if int(_read_field(fields, 'SWIZ')) != register.component: raise ValueError(f'invalid {field} SWIZ')
  return register

def _read_const(fields: Iterator[tuple[str, str | int]], half: bool) -> IR3Register:
  encoded = int(_read_field(fields, 'SRC'))
  number = int(_read_field(fields, 'CONST'))
  component = int(_read_field(fields, 'SWIZ'))
  if encoded != number * 4 + component:
    raise ValueError('invalid constant register encoding')
  return IR3Register('hc' if half else 'c', number, component)

def _read_source(fields: Iterator[tuple[str, str | int]], slot: int) -> IR3Source:
  field = f'SRC{slot}'
  encoded = int(_read_field(fields, field))
  tag = (encoded >> 11) & 0x7

  last = bool(_read_field(fields, 'LAST')) if tag == 0 else False
  absneg = int(_read_field(fields, 'ABSNEG'))
  repeat = bool(_read_field(fields, 'SRC_R'))
  value: IR3Register | int | float

  match tag:
    case 0:
      half = bool(_read_field(fields, 'HALF'))
      value = _read_gpr(fields, 'SRC', half)
      mask = 0xff
      if encoded & mask != value.number * 4 + value.component:
        raise ValueError(f'invalid {field} encoding')

    case 2 | 6:
      half = bool(_read_field(fields, 'HALF'))
      value = _read_const(fields, half)
      mask = 0x7ff
      if encoded & mask != value.number * 4 + value.component:
        raise ValueError(f'invalid {field} encoding')

    case 4:
      value = int(_read_field(fields, 'IMMED'))
      if value & 0x400: value -= 0x800

    case _:
      raise NotImplementedError(f'unsupported IR3 source encoding {encoded:#x}')

  return IR3Source(
    value=value,
    repeat=repeat,
    absolute=bool(absneg & 2),
    negate=bool(absneg & 1),
    last=last,
  )

def _decode_instruction(category: int, raw_fields: list[tuple[str, str | int]]) -> IR3Instruction:
  if category != 2: raise NotImplementedError(f'unsupported IR3 category {category}')
  fields = iter(raw_fields)
  sy = bool(_read_field(fields, 'SY'))
  ss = bool(_read_field(fields, 'SS'))
  jp = bool(_read_field(fields, 'JP'))
  sat = bool(_read_field(fields, 'SAT'))
  repeat = int(_read_field(fields, 'REPEAT'))
  ul = bool(_read_field(fields, 'UL'))
  name = str(_read_field(fields, 'NAME'))
  ei = bool(_read_field(fields, 'EI'))
  dst_half = bool(_read_field(fields, 'DST_HALF'))
  dst = _read_gpr(fields, 'DST', dst_half)

  source_count = sum(name in ('SRC1', 'SRC2', 'SRC3', 'SRC4') for name, _ in raw_fields)
  srcs = tuple(_read_source(fields, slot) for slot in range(1, source_count + 1))

  return IR3Instruction(
    name=name,
    dst=dst,
    srcs=srcs,
    sy=sy,
    ss=ss,
    jp=jp,
    sat=sat,
    repeat=repeat,
    ul=ul,
    ei=ei,
  )

def decode_ir3(code: bytes) -> list[IR3Instruction]:
  if len(code) % 8: raise ValueError('IR3 code size must be a multiple of 8 bytes')
  fields: list[tuple[str, str | int]] = []
  decoded_fields: list[list[tuple[str, str | int]]] = []

  @ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p)
  def begin_instruction(_data, _number, _instruction):
    fields.clear()

  @ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(mesa.struct_isa_decode_value))
  def collect_field(_data, name, value):
    decoded_value = value.contents
    field_name = ctypes.string_at(name).decode()
    field_value = ctypes.string_at(decoded_value.str).decode() if decoded_value.str else decoded_value.num
    fields.append((field_name, field_value))

  @ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p)
  def end_instruction(_data, _number, _instruction):
    decoded_fields.append(fields.copy())

  options = mesa.struct_isa_decode_options(630, True, 0, False, field_cb=collect_field, pre_instr_cb=begin_instruction, post_instr_cb=end_instruction)
  fp = libc.tmpfile()
  if not fp: raise OSError('failed to create IR3 disassembly stream')

  try:
    mesa_fp = ctypes.cast(fp, ctypes.POINTER(mesa.struct__IO_FILE))
    mesa.ir3_isa_disasm(code, len(code), mesa_fp, options)
  finally:
    libc.fclose(fp)

  instructions = []
  for offset, raw_fields in zip(range(0, len(code), 8), decoded_fields, strict=True):
    word = int.from_bytes(code[offset:offset + 8], 'little')
    category = (word >> 61) & 0x7
    instructions.append(_decode_instruction(category, raw_fields))
  return instructions