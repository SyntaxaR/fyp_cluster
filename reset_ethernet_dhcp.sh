nmcli con delete eth0-controller-static
nmcli con add type ethernet ifname eth0 con-name eth0-dhcp ipv4.method auto
nmcli con up eth0-dhcp