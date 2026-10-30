import sys, time
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_

def set_hands(right, left, secs=1.5):
    """0.0 closed .. 1.0 open (Inspire hands via rt/inspire/cmd)"""
    pub = ChannelPublisher("rt/inspire/cmd", MotorCmds_); pub.Init()
    m = MotorCmds_(); m.cmds = [unitree_go_msg_dds__MotorCmd_() for _ in range(12)]
    end = time.time() + secs
    while time.time() < end:
        for i in range(6):
            m.cmds[i].q = right
            m.cmds[6 + i].q = left
        pub.Write(m)
        time.sleep(0.02)

if __name__ == "__main__":
    ChannelFactoryInitialize(0, sys.argv[1])
    set_hands(float(sys.argv[2]), float(sys.argv[3]))
