#include <LPC17xx.H>

void clock_wise(void);

void anti_clock_wise(void);

unsigned long int var1;
unsigned int i, k;

int main(void) {
    SystemInit();
    SystemCoreClockUpdate();
    LPC_PINCON->PINSEL0 = 0xFFFF00FF;
    LPC_GPIO0->FIODIR |= 0x000000F0;
    LPC_GPIO2->FIODIR &= ~(1 << 12);

    while (1) {
        if (LPC_GPIO2->FIOPIN & (1 << 12)) {
            clock_wise();
        } else {
            anti_clock_wise();
        }
    }
}

void clock_wise(void) {
    var1 = 0x00000008; 
    for (i = 0; i <= 3; i++) {
        var1 = var1 << 1;
        LPC_GPIO0->FIOPIN = var1;
        for (k = 0; k < 3000; k++);
    }
}

void anti_clock_wise(void) {
    var1 = 0x00000100; 
    for (i = 0; i <= 3; i++) {
        var1 = var1 >> 1;
        LPC_GPIO0->FIOPIN = var1;
        for (k = 0; k < 3000; k++);
    }
}