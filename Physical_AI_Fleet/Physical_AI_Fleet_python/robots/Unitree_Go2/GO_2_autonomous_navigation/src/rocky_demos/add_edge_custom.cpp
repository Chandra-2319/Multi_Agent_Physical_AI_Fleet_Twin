#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/ros2_idl/QtEdge_.hpp>
#include <cstdlib>
#define TOPIC "rt/qt_add_edge"
using namespace unitree::robot;

int main(int argc, const char **argv)
{
    if (argc < 5)
    {
        std::cout << "Usage: " << argv[0] << " networkInterface edge_name start_node end_node" << std::endl;
        exit(-1);
    }
    ChannelFactory::Instance()->Init(0, argv[1]);
    ChannelPublisherPtr<unitree_interfaces::msg::dds_::QtEdge_> publisher =
        ChannelPublisherPtr<unitree_interfaces::msg::dds_::QtEdge_>(new ChannelPublisher<unitree_interfaces::msg::dds_::QtEdge_>(TOPIC));
    publisher->InitChannel();
    unitree_interfaces::msg::dds_::QtEdge_ send_msg;

    int edge_name = atoi(argv[2]);
    int start_node = atoi(argv[3]);
    int end_node = atoi(argv[4]);

    send_msg.seq_().data() = "index:123;";
    send_msg.edge_().edge_name_().push_back(edge_name);
    send_msg.edge_().start_node_name_().push_back(start_node);
    send_msg.edge_().end_node_name_().push_back(end_node);
    send_msg.edge_().dog_speed_().push_back(0.5);
    send_msg.edge_().edge_state_2_().push_back(3);
    send_msg.edge_().dog_stats_().push_back(0);
    send_msg.edge_().edge_length_().push_back(0);
    send_msg.edge_().dog_back_stats_().push_back(0);
    send_msg.edge_().edge_state_().push_back(0);
    send_msg.edge_().edge_state_1_().push_back(0);
    send_msg.edge_().edge_state_3_().push_back(0);
    send_msg.edge_().edge_state_4_().push_back(0);

    publisher->Write(send_msg);
    std::cout << "Added edge " << edge_name << ": " << start_node << " -> " << end_node << std::endl;
    sleep(2);
    return 0;
}
