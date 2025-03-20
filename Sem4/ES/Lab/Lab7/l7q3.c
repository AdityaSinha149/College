#include <LPC17xx.h>
#include <math.h>

unsigned char tohex[16] = {
    0X3F, 0X06, 0X5B, 0X4F, 0X66, 0X6D, 0X7D, 0X07, 
    0X7F, 0X6F, 0X77, 0X7C, 0X39, 0X5E, 0X79, 0X71
};

long int counter = 0;
unsigned int temp;
unsigned int j = 0,i;

void configure_7seg() {
    LPC_GPIO0->FIODIR |= 0XFF0;  // Set P0.4 - P0.11 as output for 7-segment
    LPC_GPIO1->FIODIR |= 0XF << 23;  // Set P1.23 - P1.26 as output for digit selection
}

void configure_SW2() {
    LPC_PINCON->PINSEL0 &= 0; // Clear bits 24-25 (P2.12 as GPIO)
    LPC_GPIO0->FIODIR &= 0xFFDFFFFF;   // Set P2.12 as input
}

int read_SW2() {
    return (LPC_GPIO0->FIOPIN & (1 << 21)) ? 1 : 0;
}

void delay() {
    for (j = 0; j < 10000; j++);
}

int main() {
    SystemInit();
    SystemCoreClockUpdate();
    configure_7seg();
    configure_SW2();

    while (1) {
        if (read_SW2()) {
            if (counter < 0xFFFF) counter++;
        } else {
            if (counter > 0) counter--;
        }
        
        temp = counter;
        for (i = 0; i < 4; i++) {
            LPC_GPIO1->FIOPIN = i << 23;
            LPC_GPIO0->FIOCLR = 0X00000FF0;
            LPC_GPIO0->FIOPIN = tohex[temp%16] << 4; // Extract hex digit
            temp/=16;
            for(j=0;j<1000;j++);
        }
    }
    return 0;
}
