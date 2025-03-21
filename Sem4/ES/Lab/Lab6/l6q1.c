#include <lpc17xx.h>

unsigned long led = 0x00000010;
unsigned int j;

int main() {
    SystemInit();
    SystemCoreClockUpdate();

    LPC_PINCON->PINSEL0 &= 0XFF0000FF; //not important
    LPC_GPIO0->FIODIR |= 0X00000FF0;

    while(1) {
        LPC_GPIO0->FIOCLR = 0x00000FF0;
        LPC_GPIO0->FIOSET = led;

        led += 1;
			
        if (led == 0x000001000)
            led = 0x00000010;

        for (j = 0; j < 1000; j++);
    }
    return 0;
}
