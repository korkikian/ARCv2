#!/usr/bin/env python3
"""Semantic regression tests for the ARCv2 SLEIGH module, run in Ghidra's p-code emulator.

Each test loads a few hand-encoded ARCv2 instructions (encodings from GNU binutils
opcodes/arc-tbl.h templates; the disassembly is printed so it can be checked), steps them
with ghidra.pcode.emu.PcodeEmulator and compares registers/memory with the ARCv2 ISA.

Usage (requires pyghidra and a Ghidra install, GHIDRA_INSTALL_DIR set):
    python test_semantics.py                      # use the installed ARCv2:LE:32:default
    python test_semantics.py /path/to/languages   # load ARCv2.ldefs from a directory instead
                                                  # (the directory must contain a compiled ARCv2.sla)
Exit status is the number of failed checks.
"""
import struct, sys
import pyghidra
pyghidra.start(verbose=False)
import jpype
from generic.jar import ResourceFile
from java.io import File
from ghidra.app.plugin.processors.sleigh import SleighLanguageProvider
from ghidra.program.model.lang import LanguageID
from ghidra.program.util import DefaultLanguageService
from ghidra.pcode.emu import PcodeEmulator, SleighInstructionDecoder
from ghidra.pcode.exec import PcodeExecutorStatePiece

LANG_ID = LanguageID("ARCv2:LE:32:default")
if len(sys.argv) > 1:
    ctor = SleighLanguageProvider.class_.getDeclaredConstructor(ResourceFile)
    ctor.setAccessible(True)
    provider = ctor.newInstance(ResourceFile(File(sys.argv[1] + "/ARCv2.ldefs")))
    lang = provider.getLanguage(LANG_ID)
else:
    lang = DefaultLanguageService.getLanguageService().getLanguage(LANG_ID)
ram = lang.getDefaultSpace()
aux = lang.getAddressFactory().getAddressSpace("AUX_REGS_BASE")
JB = jpype.JArray(jpype.JByte)
INSPECT = PcodeExecutorStatePiece.Reason.INSPECT

def h16(v):  # one 16-bit instruction (little endian)
    return struct.pack("<H", v)
def w32(v):  # one 32-bit instruction: high halfword first, each halfword little endian
    return struct.pack("<HH", v >> 16, v & 0xFFFF)

failures = 0

class Machine:
    def __init__(self, code, base=0x1000):
        self.emu = PcodeEmulator(lang)
        self.base = base
        self.write(base, code)
        self.t = self.emu.newThread()
        self.t.overrideCounter(ram.getAddress(base))
        self.st = self.t.getState()
        self.decoder = SleighInstructionDecoder(lang, self.emu.getSharedState())
    def write(self, addr, data):
        self.emu.getSharedState().setVar(ram.getAddress(addr), len(data), True, JB(data))
    def read32(self, addr):
        return struct.unpack("<I", bytes(self.emu.getSharedState().getVar(ram.getAddress(addr), 4, True, INSPECT)))[0]
    def reg(self, name):
        r = lang.getRegister(name)
        return int.from_bytes(bytes(self.st.getVar(r, INSPECT)), "little")
    def set(self, name, v):
        r = lang.getRegister(name)
        self.st.setVar(r, JB(int(v & ((1 << (8 * r.getNumBytes())) - 1)).to_bytes(r.getNumBytes(), "little")))
    def set_aux(self, num, v):
        self.emu.getSharedState().setVar(aux.getAddress(num * 4), 4, True, JB(struct.pack("<I", v)))
    def pc(self):
        return self.t.getCounter().getOffset()
    def disasm(self, addr):
        return str(self.decoder.decodeInstruction(ram.getAddress(addr), self.t.getContext()))
    def step(self, n=1):
        global failures
        for _ in range(n):
            try:
                self.t.stepInstruction()
            except Exception as e:
                failures += 1
                print(f"  [FAIL] could not execute the instruction at {self.pc():#x}: {str(e).splitlines()[0][:120]}")
                return False
        return True

def check(name, got, want):
    global failures
    ok = got == want
    failures += 0 if ok else 1
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: got {got:#x}, want {want:#x}" if isinstance(got, int) else
          f"  [{'PASS' if ok else 'FAIL'}] {name}: got {got}, want {want}")

def listing(m, addrs):
    for a in addrs:
        try: print(f"    {a:#06x}: {m.disasm(a)}")
        except Exception as e: print(f"    {a:#06x}: <decode error {e}>")

print("1. ADD_S/MOV_S/CMP_S h,s3: s3 encodes 0..6, and 0b111 = -1 (binutils extract_simm3s)")
m = Machine(h16(0x740C) + h16(0x7624) + h16(0x774C) + h16(0x7574))
listing(m, [0x1000, 0x1002, 0x1004, 0x1006])
m.set("r1", 10); m.set("r3", 5)
m.step(4)
check("mov_s r0,4", m.reg("r0"), 4)
check("add_s r1,r1,6 (r1=10)", m.reg("r1"), 16)
check("mov_s r2,-1", m.reg("r2"), 0xFFFFFFFF)
check("cmp_s r3,5 (r3=5) sets Z", m.reg("Z"), 1)

print("2. LD_S b,[PCL,u10]: address = (PC & ~3) + u8*4")
m = Machine(h16(0x78E0) + h16(0xD002) + h16(0x78E0) + h16(0x78E0) + struct.pack("<I", 0x11223344))
listing(m, [0x1002])
m.step(2)
check("ld_s r0,[pcl,8] at 0x1002 loads the word at 0x1008", m.reg("r0"), 0x11223344)

print("3. JLI_S u10: PC = JLI_BASE + u10*4, BLINK = next instruction")
m = Machine(h16(0x5803))
listing(m, [0x1000])
m.set_aux(0x290, 0x2000)  # JLI_BASE
m.write(0x200C, h16(0x78E0))
m.step(1)
check("jli_s 3 with JLI_BASE=0x2000 jumps to", m.pc(), 0x200C)
check("jli_s sets blink", m.reg("blink"), 0x1002)

print("4. BRcc.d: the condition is evaluated before the delay-slot instruction runs")
#   0x1000 brne.d r0,r1,0x1008 ; 0x1004 add_s r0,r0,r1 (delay slot) ; 0x1006 mov_s r2,1 ; 0x1008 mov_s r3,1
m = Machine(w32(0x08090061) + h16(0x6038) + h16(0x714C) + h16(0x716C))
listing(m, [0x1000, 0x1004])
m.set("r0", 0); m.set("r1", 2)
m.step(1)
check("brne.d r0,r1 with r0=0,r1=2 is taken even though the slot makes r0 == r1; next pc", m.pc(), 0x1008)
check("delay slot executed (r0 = 0 + 2)", m.reg("r0"), 2)

print("5. BREQ.d executes its delay slot")
#   0x1000 breq.d r0,r0,0x1008 ; 0x1004 mov_s r2,1 (delay slot) ; 0x1006 nop_s ; 0x1008 nop_s
m = Machine(w32(0x08090020) + h16(0x714C) + h16(0x78E0) + h16(0x78E0))
listing(m, [0x1000])
m.step(1)
check("breq.d r0,r0 taken; next pc", m.pc(), 0x1008)
check("delay slot executed (r2 = 1)", m.reg("r2"), 1)

print("6. CLRI c / SETI c (cf. QEMU target/arc semfunc-v2.c)")
#   0x1000 clri r2 ; 0x1004 seti r2
m = Machine(w32(0x272F00BF) + w32(0x262F00BF))
listing(m, [0x1000, 0x1004])
m.set("IE", 1); m.set("E", 5); m.set("r2", 0x3000)
m.write(0x3000, struct.pack("<I", 0xA5A5A5A5))
m.step(1)
check("clri r2: r2 = 0x20 | IE<<4 | E", m.reg("r2"), 0x35)
check("clri r2: IE cleared", m.reg("IE"), 0)
check("clri r2: memory at old r2 untouched", m.read32(0x3000), 0xA5A5A5A5)
m.set("E", 0); m.set("r2", 0x35)
m.step(1)
check("seti r2 (r2=0x35): IE restored", m.reg("IE"), 1)
check("seti r2 (r2=0x35): E restored", m.reg("E"), 5)

print("7. RTIE does not fall through")
m = Machine(w32(0x246F003F))
listing(m, [0x1000])
ins = m.decoder.decodeInstruction(ram.getAddress(0x1000), m.t.getContext())
check("rtie has fall-through", bool(ins.getFlowType().hasFallthrough()), False)

print("8. BTST_S b,u5 tests the bit (Z set when the bit is clear)")
m = Machine(h16(0xB9E4))                       # btst_s r1,0x4
listing(m, [0x1000])
m.set("r1", 0x0F); m.set("Z", 0)
m.step(1)
check("btst_s r1,4 with r1=0x0f sets Z", m.reg("Z"), 1)

print("9. Register 63 (PCL) as a source operand")
m = Machine(h16(0x78E0) + w32(0x2740748A))     # nop_s ; add r10,pcl,0x12  (at 0x1002)
listing(m, [0x1002])
m.step(2)
check("add r10,pcl,0x12 at 0x1002 gives (0x1002 & ~3) + 0x12", m.reg("r10"), 0x1012)

print("10. SETcc a,b,c compares b with c")
m = Machine(w32(0x213A0080))                   # setlt r0,r1,r2
listing(m, [0x1000])
m.set("r1", 1); m.set("r2", 2)
m.step(1)
check("setlt r0,r1,r2 with r1=1,r2=2", m.reg("r0"), 1)

print("11. LD.AS a,[b,c] scales the value of register c")
m = Machine(w32(0x21F00080))                   # ld.as r0,[r1,r2]
listing(m, [0x1000])
m.set("r1", 0x3000); m.set("r2", 3)
m.write(0x3008, struct.pack("<I", 0x11111111)); m.write(0x300C, struct.pack("<I", 0x22222222))
m.step(1)
check("ld.as r0,[r1,r2] with r1=0x3000,r2=3 loads 0x300c", m.reg("r0"), 0x22222222)

print("12. SLEEP c with c[4] set enables interrupts and sets E")
m = Machine(w32(0x212F003F))                   # sleep r0
listing(m, [0x1000])
m.set("r0", 0x15); m.set("IE", 0); m.set("E", 0)
m.step(1)
check("sleep r0 (r0=0x15): IE", m.reg("IE"), 1)
check("sleep r0 (r0=0x15): E", m.reg("E"), 5)

print(f"\n{failures} check(s) failed")
sys.exit(failures)
