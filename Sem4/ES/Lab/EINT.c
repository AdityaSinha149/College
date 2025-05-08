#include <lpc17xx.h>   

void EINT0_IRQHandler(void)
{
    LPC_SC->EXTINT = (1 << 0);           /* Clear Interrupt Flag */
    LPC_GPIO2->FIOPIN ^= (1 << 0);       /* Toggle LED1 (P2.0) on EINT0 */
}

void EINT1_IRQHandler(void)
{
    LPC_SC->EXTINT = (1 << 1);           /* Clear Interrupt Flag */
    LPC_GPIO2->FIOPIN ^= (1 << 1);       /* Toggle LED2 (P2.1) on EINT1 */
}

int main()
{
    SystemInit();

    LPC_SC->EXTINT      = (1 << 0) | (1 << 1);               /* Clear Pending interrupts */
    LPC_PINCON->PINSEL4 = (1 << 20) | (1 << 22);             /* Configure P2.10, P2.11 as EINT0/1 */
    LPC_SC->EXTMODE     = (1 << 0) | (1 << 1);               /* Configure EINT0/1 as Edge Triggered */
    LPC_SC->EXTPOLAR    = (1 << 0) | (1 << 0);               /* Configure EINT0/1 as Falling Edge */

    LPC_GPIO2->FIODIR   = (1 << 0) | (1 << 1);               /* Set P2.0 and P2.1 as OUTPUT (LEDs) */
    LPC_GPIO2->FIOPIN   = 0x00;                              /* Turn off LEDs */

    NVIC_EnableIRQ(EINT0_IRQn);                             /* Enable EINT0 interrupt */
    NVIC_EnableIRQ(EINT1_IRQn);                             /* Enable EINT1 interrupt */

    while(1)
    {
        // Idle loop
    }
}
