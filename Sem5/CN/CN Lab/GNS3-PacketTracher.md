- ?, help is the man of gns3
[]()# PC
```bash
> ip <ip-address>/<subnet> <gateway-address>
```
- use command 'show' or 'show ip' to see pc configurations
# Router
```bash
> en
> conf t
> int <interface-name>
> ip address <ip> <subnet-dot-format>
> no shut
```
### Static Routing
```bash
> en
> conf t
> ip route <destination-network> <subnet-mask> <next-hop-address>
# next-hop-address is the gateway address of the next router to which the packet should be sent if u want it to go to destination-network
> end
```
- use command 'show ip route' to see all routes set and 'show ip route static' to see all static routes set
- use command 'show ip interface brief' to see all int's ip settings, 'show interface' or 'show ip interface' to see it in detail
### Dynamic Routing
```bash
> en
> conf t
> router rip
> version 2
> #advertise every network
> network <ip> <subnet-dotted-notation>
> no auto-summary
```
# VLAN
### Router Part
```bash
> en
> conf t
> int <interface-name>
> no shut
> 
> #do this for every vlan connected to that interface
> int <interface-name>.<sub-interface-number>
> encapsulation dot1Q <VLAN-number>
> ip address <ip> <subnet-dot-format>
```
### Switch Part
```bash
> #create every vlan
> vlan <vlan-number>
> name <vlan-name>
> 
> #assign switchports to vlan
> int range fa0/1 - 18 or int range fa0/1, fa0/2, fa0/5, fa0/18
> switchport mode access
> switchport access vlan 4
> 
> #make trunk ports
> int fa0/1
> switchport trunk encapsulation dot1q
> switchport mode trunk
> switchport trunk allowed vlan 10,20...
> switchport trunk native vlan 99
> no shut
```
# DHCP
```bash
> en
> conf t
> ip dhcp exclude-address 192.168.1.1 192.168.1.15
> 
> #make dhcp pools
> ip dhcp pool <pool-name>
> #assign pool defaults
> network <network-address> <subnet-dot-notation>
> default-router <gateway-address>
> dns-server <dns-ip>
```
- use command 'show ip dhcp pool' to see pool data and 'show run | include excluded' to see excluded ips
# DNS
```bash
> ip dns server
> ip domain-name <router-domain-name>
> #set every dns entry
> ip host <url/local dns> <ip>
```
- use command 'show hosts' to see dns entries