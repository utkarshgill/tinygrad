from test.mockgpu.qcom.emu import IR3Instruction, IR3Register, IR3Source, decode_ir3

def test_decode_add():
  expected = IR3Instruction(
    name='add.f',
    dst=IR3Register('r', 2, 3),
    srcs=(
      IR3Source(IR3Register('r', 2, 3)),
      IR3Source(IR3Register('r', 4, 0)),
    ),
    sy=True,
  )
  assert decode_ir3(bytes.fromhex('0b0010000b001050')) == [expected]
