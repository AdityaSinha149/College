#include <LPC17xx.h>

void configureSW2(){
    LPC_PINCON->PINSEL3 &= ~(3<<20); // Make P1.26 GPIO
    LPC_GPIO1->FIODIR &= ~(1<<26); // Set P1.26 as input
}

void configureLED(){
    LPC_PINCON->PINSEL0 &= FFFFF00F; // Make P0.4 to P0.11 GPIO
    LPC_GPIO0->FIODIR |= (0xFF<<4); // Set P0.4 to P0.11 as output 
}

int readSW2(){
    return (LPC_GPIO1->FIOPIN & (1<<26)) == 0; // Read P1.26
}

int main(){
    int i = 0, j = 0;
    configureSW2(); // Configure SW2
    configureLED(); // Configure LEDs

    while(1){
        if(readSW2()){ // If SW2 is pressed
            if(i == 0 || i == 7) LPC_GPIO0->FIOPIN = 0xFF << 4; // 11111111
            if(i == 1 || i == 6) LPC_GPIO0->FIOPIN = 0x7E << 4; // 01111110
            if(i == 2 || i == 5) LPC_GPIO0->FIOPIN = 0x3C << 4; // 00111100
            if(i == 3 || i == 4) LPC_GPIO0->FIOPIN = 0x18 << 4; // 00110000
            i = (i+1) % 8; // Increment i and wrap around
            for(j = 0; j < 5000; j++); // Delay
        }
        else{
            LPC_GPIO0->FIOPIN = 0x00 << 4; // Turn off all LEDs
            i = 0; // Reset i
        }
    }
    return 0;
}