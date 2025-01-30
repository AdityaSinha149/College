	AREA RESET, DATA, READONLY
		EXPORT __Vectors

__Vectors
	dcd 0x40001000
	dcd Reset_Handler
	align
	AREA mycode, CODE, READONLY
	ENTRY
	EXPORT Reset_Handler

Reset_Handler
	ldr r6, =result
	mov r2, #0
	ldr r0, =value1
	ldr r1, [r0]
up	cmp r1, #0xa
	bcc store
	sub r1, #0xa
	add r2, #0x01
	b up
	
store 
	strb r1, [r6], #1
	mov r1, r2
	mov r2, #0
	cmp r1, #0xa
	bcs up
	strb r1, [r6], #1
	mov r3, r6
	ldr r2, =packed
	ldr r6, =result
	sub r4, r3, r6; no. of unpacked numbers
up1	ldrb r0, [r6], #1
	ldrb r1, [r6], #1
	lsl r1, #4
	orr r1, r0
	strb r1, [r2], #1
	sub r4, #2
	cmp R4, #1
	BGE up1
	
stop b stop

value1 dcd 0x04d2
	area data, data,readwrite
result dcd 0,0,0,0,0,0,0,0,0,0
packed dcd 0
		end 