	AREA RESET,DATA,READONLY
	EXPORT __Vectors
__Vectors
	DCD 0X10001000
	DCD Reset_Handler
	ALIGN
	AREA mycode,CODE,READONLY
	ENTRY
	EXPORT Reset_Handler
Reset_Handler
	LDR R0,SRC
	MOV R1,#0xF
	MOV R3,#0xF0000
L1	AND R2,R0,R1
	ADD R7,R7,R2
	LSR R0,#4
	LSL R1,#4
	SUBS R4,R3,R1
	BNE L1
	LDR R0,=DST
	STR R7,[R0]
STOP B STOP
SRC DCD 0x01020304
	AREA mydata, DATA, READWRITE
DST DCD 0
	END