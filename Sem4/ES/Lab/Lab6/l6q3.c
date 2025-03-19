#include <lpc17xx.h>

unsigned int counter = 0x00000001;
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
            LPC_GPIO0->FIOCLR = 0x00000FF0;
            LPC_GPIO0->FIOSET = counter;
            counter=counter<<4;
            if (counter == 0x10000000)
                counter = 0x00000001;
        }
        for (j = 0; j < 1000; j++);
    }

    return 0;
}
