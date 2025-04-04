#include <lpc17xx.h>

unsigned int counter = 0x00000010;
unsigned int j;

void configure_SW2() {
    LPC_PINCON->PINSEL4 &= (0xFCFFFFFF); //not important
    LPC_GPIO2->FIODIR &= 0xFFFFFFFF;
}

void configure_LEDs() {
    LPC_PINCON->PINSEL0 &= 0xFF0000FF; //not important
    LPC_GPIO0->FIODIR |= 0x00000FF0;
}

int read_SW2() {
    return (LPC_GPIO2->FIOPIN & (1 << 12)) ? 1 : 0;
}

int main() {
    SystemInit();
    SystemCoreClockUpdate();

    configure_SW2();
    configure_LEDs();

    while (1) {
        if (read_SW2()) {
            if (counter < 0x00001000) {
                counter++;
            }
        } else {
            if (counter > 0) {
                counter--;
            }
        }

        LPC_GPIO0->FIOPIN = counter;

        for (j = 0; j < 1000; j++);
    }

    return 0;
}
