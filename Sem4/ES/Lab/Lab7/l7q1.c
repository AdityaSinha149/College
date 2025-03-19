#include <lpc17xx.h>

unsigned char tohex[4] = {0x06, 0x5B, 0x4F, 0x66};
unsigned int pos={0<<23,1<<23,2<<23,3<<23};

void configure_7seg() {
    LPC_GPIO0->FIODIR |= 0XFF0;
    LPC_GPIO1->FIODIR |= 0XF << 23;
}

int main()
{
    SystemInit();
    SystemCoreClockUpdate();
    configure_7seg();

    while (1)
    {
        for (int i = 0; i < 4; i++)
        {
            LPC_GPIO0->FIOPIN = pos[i];
            LPC_GPIO1->FIOPIN = tohex[4-i];
            for (int j = 0; j < 1000; j++);
        }
    }
    return 0;
}