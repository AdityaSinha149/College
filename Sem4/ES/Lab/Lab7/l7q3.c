#include <LPC17xx.h>
#include <math.h>

unsigned char tohex[10] = {0X3F, 0X06, 0X5B, 0X4F, 0X66, 0X6D, 0X7D, 0X07,0X7F, 0X6F};

unsigned int counter=0;

unsigned int j = 0;

void configure_7seg() {
    LPC_GPIO0->FIODIR |= 0XFF0;
    LPC_GPIO1->FIODIR |= 0XF << 23;
}

void configure_SW2() {
    LPC_PINCON->PINSEL4 &= (0xFCFFFFFF); //not important
    LPC_GPIO2->FIODIR &= 0xFFFFFFFF;
}

int read_SW2() {
    return (LPC_GPIO2->FIOPIN & (1 << 12)) ? 1 : 0;
}

int main()
{
    SystemInit();
    SystemCoreClockUpdate();
    configure_7seg();
    configure_SW2();

    while(1){
        if (read_SW2())
            if (counter < 9999)
                counter++;
        else
            if (counter > 0)
                counter--;

        unsigned int temp=counter;
        for(int i=0;i<4;i++){
            LPC_GPIO1->FIOPIN = i << 23;
            LPC_GPIO0->FIOCLR |= 0X00000FF0;
            LPC_GPIO0->FIOPIN = tohex[temp%10] << 4;
            temp/=10;
            for (j = 0; j < 1000; j++);        
        }

        for (j = 0; j < 1000; j++);
    }
    return 0;
}