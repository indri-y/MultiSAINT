`celldefine
// LUTs
module LUT1 (O, I0); output O; input I0; endmodule
module LUT2 (O, I0, I1); output O; input I0, I1; endmodule
module LUT3 (O, I0, I1, I2); output O; input I0, I1, I2; endmodule
module LUT4 (O, I0, I1, I2, I3); output O; input I0, I1, I2, I3; endmodule
module LUT5 (O, I0, I1, I2, I3, I4); output O; input I0, I1, I2, I3, I4; endmodule
module LUT6 (O, I0, I1, I2, I3, I4, I5); output O; input I0, I1, I2, I3, I4, I5; endmodule

// FFs
module FDCE (Q, C, CE, CLR, D); output Q; input C, CE, CLR, D; endmodule
module FDPE (Q, C, CE, D, PRE); output Q; input C, CE, D, PRE; endmodule
// (opsional) FDRE/FDSE jika dataset lain punya
module FDRE (Q, C, CE, D, R); output Q; input C, CE, D, R; endmodule
module FDSE (Q, C, CE, D, S); output Q; input C, CE, D, S; endmodule

// Carry & mux
module CARRY4 (CO, O, CI, CYINIT, DI, S); output [3:0] CO, O; input CI, CYINIT; input [3:0] DI, S; endmodule
module MUXCY (O, CI, DI, S); output O; input CI, DI, S; endmodule
module XORCY (O, CI, LI); output O; input CI, LI; endmodule
module MUXF7 (O, I0, I1, S); output O; input I0, I1, S; endmodule
module MUXF8 (O, I0, I1, S); output O; input I0, I1, S; endmodule

// IO & global & const
module BUFG (O, I); output O; input I; endmodule
module IBUF (O, I); output O; input I; endmodule
module OBUF (O, I); output O; input I; endmodule
module VCC (P); output P; endmodule
module GND (G); output G; endmodule

// SRL
module SRL16E (Q, A0, A1, A2, A3, CE, CLK, D); output Q; input A0, A1, A2, A3, CE, CLK, D; endmodule

// BRAM (stub ringkas)
module RAMB18E1 (DOADO, DOBDO, ADDRARDADDR, ADDRBWRADDR, CLKARDCLK, CLKBWRCLK,
                 ENARDEN, ENBWREN, DIADI, DIBDI, DIPADIP, DIPBDIP, WEA, WEBWE,
                 RSTRAMARSTRAM, RSTRAMB, RSTREGARSTREG, RSTREGB, REGCEAREGCE, REGCEB,
                 CASCADEINA, CASCADEINB, CASCADEOUTA, CASCADEOUTB);
  output [15:0] DOADO, DOBDO; output CASCADEOUTA, CASCADEOUTB;
  input [13:0] ADDRARDADDR, ADDRBWRADDR; input CLKARDCLK, CLKBWRCLK, ENARDEN, ENBWREN;
  input [15:0] DIADI, DIBDI; input [1:0] DIPADIP, DIPBDIP, WEA; input [3:0] WEBWE;
  input RSTRAMARSTRAM, RSTRAMB, RSTREGARSTREG, RSTREGB, REGCEAREGCE, REGCEB, CASCADEINA, CASCADEINB;
endmodule

module RAMB36E1 (DOADO, DOBDO, ADDRARDADDR, ADDRBWRADDR, CLKARDCLK, CLKBWRCLK,
                 ENARDEN, ENBWREN, DIADI, DIBDI, DIPADIP, DIPBDIP, WEA, WEBWE,
                 RSTRAMARSTRAM, RSTRAMB, RSTREGARSTREG, RSTREGB, REGCEAREGCE, REGCEB);
  output [31:0] DOADO, DOBDO; 
  input [15:0] ADDRARDADDR, ADDRBWRADDR; input CLKARDCLK, CLKBWRCLK, ENARDEN, ENBWREN;
  input [31:0] DIADI, DIBDI; input [3:0] DIPADIP, DIPBDIP, WEA; input [7:0] WEBWE;
  input RSTRAMARSTRAM, RSTRAMB, RSTREGARSTREG, RSTREGB, REGCEAREGCE, REGCEB;
endmodule

// DSP (opsional)
module DSP48E1 (P, A, B, C, D, CLK, CARRYIN, ALUMODE, OPMODE, CARRYINSEL, INMODE,
                CEA1, CEA2, CEB1, CEB2, CEC, CED, CEP, RSTA, RSTB, RSTC, RSTD, RSTP);
  output [47:0] P; input [29:0] A; input [17:0] B; input [47:0] C; input [24:0] D;
  input CLK, CARRYIN; input [3:0] ALUMODE; input [6:0] OPMODE; input [2:0] CARRYINSEL; input [4:0] INMODE;
  input CEA1, CEA2, CEB1, CEB2, CEC, CED, CEP; input RSTA, RSTB, RSTC, RSTD, RSTP;
endmodule
`endcelldefine
