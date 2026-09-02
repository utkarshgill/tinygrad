import ctypes, functools, struct
from collections.abc import Iterator
from dataclasses import dataclass

from tinygrad import dtypes
from tinygrad.codegen import to_program
from tinygrad.device import Buffer, Device
from tinygrad.engine.realize import get_runtime
from tinygrad.helpers import Context
from tinygrad.uop.ops import AddrSpace, KernelInfo, UOp
from tinygrad.runtime.autogen import libc, mesa

# IR3 float-immediate lookup table from Mesa ir3-common.xml #flut.
_FLOAT_IMMEDIATE_BITS = (
  0x00000000, 0x3f000000, 0x3f800000, 0x40000000,
  0x402df854, 0x40490fdb, 0x3ea2f983, 0x3f317218,
  0x3fb8aa3b, 0x3e9a209b, 0x40549a78, 0x40800000,
)

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
  nop: int = 0
  ul: bool = False
  ei: bool = False

class WaveState:
  def __init__(self, fiber_count: int, wave_size: int):
    if not 0 < fiber_count <= wave_size:
      raise ValueError(f'invalid fiber count {fiber_count}')

    self.fiber_count = fiber_count
    self.wave_size = wave_size

    self.r_buf = Buffer('CPU', 64 * 4 * wave_size, dtypes.uint32).ensure_allocated()
    self._r = self.r_buf.as_memoryview(force_zero_copy=True).cast('I')

    self.hr_buf = Buffer('CPU', 64 * 4 * wave_size, dtypes.uint16).ensure_allocated()
    self._hr = self.hr_buf.as_memoryview(force_zero_copy=True).cast('H')

  def _register_index(self, register: IR3Register, fiber: int, kind: str) -> int:
    if register.kind != kind:
      raise ValueError(f'expected {kind} register, got {register.kind}')
    if not 0 <= register.number < 64 or register.number in (61, 62):
      raise ValueError(f'invalid {kind} register number {register.number}')
    if not 0 <= register.component < 4:
      raise ValueError(f'invalid register component {register.component}')
    if not 0 <= fiber < self.fiber_count:
      raise ValueError(f'invalid fiber {fiber}')
    return (register.number * 4 + register.component) * self.wave_size + fiber

  def write_r(self, register: IR3Register, fiber: int, value: int):
    self._r[self._register_index(register, fiber, 'r')] = value & 0xffffffff

  def read_r(self, register: IR3Register, fiber: int) -> int:
    return self._r[self._register_index(register, fiber, 'r')]

  def write_hr(self, register: IR3Register, fiber: int, value: int):
    self._hr[self._register_index(register, fiber, 'hr')] = value & 0xffff

  def read_hr(self, register: IR3Register, fiber: int) -> int:
    return self._hr[self._register_index(register, fiber, 'hr')]

class _IR3UOpContext:
  def __init__(self, wave_size: int, fiber_count: int, register_kinds: tuple[str, ...]):
    self.wave_size = wave_size
    self.registers = {
      kind: UOp.param(slot, dtypes.uint32 if kind == 'r' else dtypes.uint16, 64 * 4 * wave_size)
      for slot, kind in enumerate(register_kinds)
    }
    self.fiber = UOp.range(fiber_count, 0, dtype=dtypes.int)

  def register_index(self, register: IR3Register) -> UOp:
    flat = register.number * 4 + register.component
    return UOp.const(flat * self.wave_size, dtypes.int) + self.fiber

  def register_buffer(self, register: IR3Register) -> UOp:
    try:
      return self.registers[register.kind]
    except KeyError:
      raise ValueError(f'unsupported UOp register kind {register.kind}') from None

  def read_register(self, register: IR3Register) -> UOp:
    return self.register_buffer(register).index(self.register_index(register)).load()

  def write_register(self, register: IR3Register, value: UOp) -> UOp:
    return self.register_buffer(register).index(self.register_index(register)).store(value)

def _apply_float_modifiers(value: UOp, absolute: bool, negate: bool) -> UOp:
  if value.dtype == dtypes.float16:
    bit_dtype, sign_bit = dtypes.uint16, 0x8000
  elif value.dtype == dtypes.float32:
    bit_dtype, sign_bit = dtypes.uint32, 0x80000000
  else: raise ValueError(f'unsupported float modifier dtype {value.dtype}')

  bits = value.bitcast(bit_dtype)
  if absolute: bits = bits & UOp.const(sign_bit - 1, bit_dtype)
  if negate: bits = bits ^ UOp.const(sign_bit, bit_dtype)
  return bits.bitcast(value.dtype)

def _compile_runner(store: UOp, fiber: UOp, name: str):
  sink = UOp.sink(store.end(fiber)).replace(arg=KernelInfo(name=name)).rtag(1)
  with Context(NOOPT=1, PROFILE=0):
    program = to_program(sink, Device['CPU'].renderer)
    return get_runtime('CPU', program)

@functools.cache
def _add_f_runner(wave_size: int, fiber_count: int, instruction: IR3Instruction):
  if instruction.dst is None or len(instruction.srcs) != 2:
    raise ValueError('add.f requires one destination and two sources')

  dst = instruction.dst
  src0, src1 = instruction.srcs
  if not isinstance(src0.value, IR3Register) or not isinstance(src1.value, IR3Register):
    raise NotImplementedError('add.f currently requires register sources')
  if dst.kind not in ('r', 'hr') or src0.value.kind not in ('r', 'hr') or src1.value.kind != src0.value.kind:
    raise NotImplementedError('add.f currently requires matching source register widths')

  source_dtype = dtypes.float16 if src0.value.kind == 'hr' else dtypes.float32
  destination_dtype = dtypes.float16 if dst.kind == 'hr' else dtypes.float32
  destination_bits = dtypes.uint16 if dst.kind == 'hr' else dtypes.uint32
  register_kinds = tuple(kind for kind in ('r', 'hr') if kind in (dst.kind, src0.value.kind))

  context = _IR3UOpContext(wave_size, fiber_count, register_kinds)
  src0_value = _apply_float_modifiers(
    context.read_register(src0.value).bitcast(source_dtype), src0.absolute, src0.negate)
  src1_value = _apply_float_modifiers(
    context.read_register(src1.value).bitcast(source_dtype), src1.absolute, src1.negate)

  result = src0_value + src1_value
  if source_dtype == dtypes.float16 and destination_dtype == dtypes.float32:
    rounded_bits = UOp.placeholder((fiber_count,), dtypes.uint16, slot=0, addrspace=AddrSpace.REG)
    rounding_store = rounded_bits.index(context.fiber).store(result.bitcast(dtypes.uint16))
    result = rounded_bits.after(rounding_store).index(context.fiber).load().bitcast(dtypes.float16)

  result = result.cast(destination_dtype).bitcast(destination_bits)
  store = context.write_register(dst, result)
  return _compile_runner(store, context.fiber, 'ir3_add_f'), register_kinds

def execute_instruction(state: WaveState, instruction: IR3Instruction):
  if instruction.name != 'add.f':
    raise NotImplementedError(f'unsupported IR3 instruction {instruction.name}')
  if instruction.dst is None or len(instruction.srcs) != 2:
    raise ValueError('add.f requires one destination and two sources')
  if instruction.sat or instruction.repeat or any(source.repeat for source in instruction.srcs):
    raise NotImplementedError('add.f saturation and repeat are not implemented')

  dst = instruction.dst
  src0_value, src1_value = (source.value for source in instruction.srcs)

  if not isinstance(dst, IR3Register) or dst.kind not in ('r', 'hr'):
    raise NotImplementedError('add.f currently requires a general-register destination')
  if not isinstance(src0_value, IR3Register) or not isinstance(src1_value, IR3Register):
    raise NotImplementedError('add.f currently requires register sources')
  if src0_value.kind not in ('r', 'hr') or src1_value.kind != src0_value.kind:
    raise NotImplementedError('add.f currently requires matching source register widths')

  runner, register_kinds = _add_f_runner(state.wave_size, state.fiber_count, instruction)
  buffers = {'r': state.r_buf, 'hr': state.hr_buf}
  runner(*(buffers[kind]._buf for kind in register_kinds))

def execute_instructions(state: WaveState, instructions: list[IR3Instruction]):
  for instruction in instructions:
    if instruction.name == 'end': return
    if instruction.name == 'nop': continue
    execute_instruction(state, instruction)
  raise ValueError('IR3 instruction stream has no end')

def _read_field(fields: Iterator[tuple[str, str | int]], expected_name: str) -> str | int:
  try: name, value = next(fields)
  except StopIteration: raise ValueError(f'missing IR3 field {expected_name}') from None
  if name != expected_name: raise ValueError(f'expected IR3 field {expected_name}, got {name}')
  return value

def _read_repeat_or_nop(fields: Iterator[tuple[str, str | int]]) -> tuple[int, int]:
  try: name, value = next(fields)
  except StopIteration: raise ValueError('missing IR3 REPEAT or NOP field') from None

  if name == 'REPEAT': return int(value), 0
  if name == 'NOP': return 0, int(value)
  raise ValueError(f'expected IR3 field REPEAT or NOP, got {name}')

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
  encoding = (encoded >> 11) & 0x7

  last = bool(_read_field(fields, 'LAST')) if encoding == 0 else False
  absneg = int(_read_field(fields, 'ABSNEG'))
  repeat = bool(_read_field(fields, 'SRC_R'))
  value: IR3Register | int | float

  match encoding:
    case 0b000:
      half = bool(_read_field(fields, 'HALF'))
      value = _read_gpr(fields, 'SRC', half)
      mask = 0xff
      if encoded & mask != value.number * 4 + value.component:
        raise ValueError(f'invalid {field} encoding')

    # Mesa defines constant sources as x10, so both 010 and 110 select a constant register.
    case 0b010 | 0b110:
      half = bool(_read_field(fields, 'HALF'))
      value = _read_const(fields, half)
      mask = 0x7ff
      if encoded & mask != value.number * 4 + value.component:
        raise ValueError(f'invalid {field} encoding')

    case 0b100:
      value = int(_read_field(fields, 'IMMED'))
      if value & 0x400: value -= 0x800

    case 0b101:
      index = int(_read_field(fields, 'IMMED'))
      if index not in range(len(_FLOAT_IMMEDIATE_BITS)):
        raise ValueError(f'invalid IR3 float immediate {index}')
      value = struct.unpack('<f', _FLOAT_IMMEDIATE_BITS[index].to_bytes(4, 'little'))[0]
      if encoded & 0x400:
        value = struct.unpack('<e', struct.pack('<e', value))[0]

    case _:
      raise NotImplementedError(f'unsupported IR3 source encoding {encoded:#x}')

  return IR3Source(
    value=value,
    repeat=repeat,
    absolute=bool(absneg & 2),
    negate=bool(absneg & 1),
    last=last,
  )

def _decode_cat0(raw_fields: list[tuple[str, str | int]]) -> IR3Instruction:
  fields = iter(raw_fields)
  sy = bool(_read_field(fields, 'SY'))
  ss = bool(_read_field(fields, 'SS'))
  eq = bool(_read_field(fields, 'EQ'))
  jp = bool(_read_field(fields, 'JP'))
  repeat = int(_read_field(fields, 'REPEAT'))
  name = str(_read_field(fields, 'NAME'))

  if name not in ('nop', 'end'):
    raise NotImplementedError(f'unsupported Cat0 instruction {name}')
  if eq:
    raise NotImplementedError('Cat0 EQ is not implemented')

  return IR3Instruction(name, None, (), sy=sy, ss=ss, jp=jp, repeat=repeat)

def _decode_instruction(category: int, raw_fields: list[tuple[str, str | int]]) -> IR3Instruction:
  if category == 0: return _decode_cat0(raw_fields)
  if category != 2: raise NotImplementedError(f'unsupported IR3 category {category}')
  fields = iter(raw_fields)
  sy = bool(_read_field(fields, 'SY'))
  ss = bool(_read_field(fields, 'SS'))
  jp = bool(_read_field(fields, 'JP'))
  sat = bool(_read_field(fields, 'SAT'))
  repeat, nop = _read_repeat_or_nop(fields)
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
    nop=nop,
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
