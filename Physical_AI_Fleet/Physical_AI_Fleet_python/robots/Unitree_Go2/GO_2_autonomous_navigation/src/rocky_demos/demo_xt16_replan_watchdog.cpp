#include "unitree/common/dds/dds_easy_model.hpp"
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

#include <unitree/ros2_idl/QtCommand_.hpp>
#include <unitree/ros2_idl/String_.hpp>
#include <unitree/ros2_idl/Odometry_.hpp>
#include <unitree/ros2_idl/QtNode_.hpp>
#include <unitree/ros2_idl/QtEdge_.hpp>

#include <stdio.h>
#include <termio.h>
#include <cmath>
#include <ctime>
#include <chrono>
#include <mutex>

#define COMMANDTOPIC "rt/qt_command"
#define NOTICETOPIC "rt/qt_notice"
#define ODOMTOPIC "rt/lio_sam_ros2/mapping/re_location_odometry"
#define ADDNODETOPIC "rt/qt_add_node"
#define ADDEDGETOPIC "rt/qt_add_edge"

#define OBSTACLE_TIMEOUT_SECONDS 15

using namespace unitree::robot;
using namespace std;
using Clock = std::chrono::steady_clock;

struct nodeAttribute
{
    u_int16_t nodeName;
    float nodeX, nodeY, nodeZ, nodeYaw;
};

struct edgeAttribute
{
    u_int16_t edgeName, edgeStart, edgeEnd;
};

class slamDemo
{
private:
    int index = 0;
    const nav_msgs::msg::dds_::Odometry_ *currentOdom;
    u_int16_t node_name = 0;

    vector<nodeAttribute> nodeAttributeList;
    vector<edgeAttribute> edgeAttributeList;

    // Watchdog state
    std::mutex watchdogMutex;
    int stuckGoalNode = -1;
    int stuckEdgeName = -1;
    Clock::time_point stuckSince;
    bool rerouteSentForThisStuck = false;

    ChannelPublisherPtr<unitree_interfaces::msg::dds_::QtCommand_> pubQtCommand = ChannelPublisherPtr<unitree_interfaces::msg::dds_::QtCommand_>(new ChannelPublisher<unitree_interfaces::msg::dds_::QtCommand_>(COMMANDTOPIC));
    ChannelPublisherPtr<unitree_interfaces::msg::dds_::QtNode_> pubQtNode = ChannelPublisherPtr<unitree_interfaces::msg::dds_::QtNode_>(new ChannelPublisher<unitree_interfaces::msg::dds_::QtNode_>(ADDNODETOPIC));
    ChannelPublisherPtr<unitree_interfaces::msg::dds_::QtEdge_> pubQtEdge = ChannelPublisherPtr<unitree_interfaces::msg::dds_::QtEdge_>(new ChannelPublisher<unitree_interfaces::msg::dds_::QtEdge_>(ADDEDGETOPIC));
    ChannelSubscriberPtr<std_msgs::msg::dds_::String_> subQtNotice = ChannelSubscriberPtr<std_msgs::msg::dds_::String_>(new ChannelSubscriber<std_msgs::msg::dds_::String_>(NOTICETOPIC));
    ChannelSubscriberPtr<nav_msgs::msg::dds_::Odometry_> subOdommetry = ChannelSubscriberPtr<nav_msgs::msg::dds_::Odometry_>(new ChannelSubscriber<nav_msgs::msg::dds_::Odometry_>(ODOMTOPIC));

public:
    slamDemo(const char *networkInterface);

    void qtNoticeHandler(const void *message);
    void odometryHandler(const void *message);

    void keyExecute();
    unsigned char keyDetection();

    void startMapping();
    void endMapping();
    void startRelocation();
    void initPose();
    void startNavigation();
    void defaultNavigation();
    void addNodeAndEdge();
    void addEdge(u_int16_t edge_name, u_int16_t start_node, u_int16_t end_node);
    void saveNodeAndEdge();
    void addManualBypassEdge();
    void closeAllNode();
    void deleteAllNode();
    void deleteAllEdge();
    void pauseNavigation();
    void recoverNavigation();
    void sendSingleNodeGoal(int target_node); // bypasses edge routing entirely
};

slamDemo::slamDemo(const char *networkInterface)
{
    ChannelFactory::Instance()->Init(0, networkInterface);

    pubQtCommand->InitChannel();
    pubQtNode->InitChannel();
    pubQtEdge->InitChannel();

    subQtNotice->InitChannel(std::bind(&slamDemo::qtNoticeHandler, this, std::placeholders::_1), 10);
    subOdommetry->InitChannel(std::bind(&slamDemo::odometryHandler, this, std::placeholders::_1), 1);

    cout << "***********************  Unitree SLAM Demo (Watchdog) ***********************\n";
    cout << "q:Close  w:StartMap  e:EndMap  a:StartNav  s:Pause  d:Recover\n";
    cout << "z:Relocate  x:AddNode  c:Save  v:Delete  b:AddBypassEdge\n";
    cout << "Watchdog: auto single-node-nav reroute after " << OBSTACLE_TIMEOUT_SECONDS << "s of the same obstacle\n";
    cout << "------------------------------------------------------------------" << endl;
}

void slamDemo::odometryHandler(const void *message)
{
    currentOdom = (const nav_msgs::msg::dds_::Odometry_ *)message;
}

// Extracts an integer value following a given label in a string, e.g. "Goal node name:3."
static int extractIntAfter(const string &text, const string &label)
{
    size_t pos = text.find(label);
    if (pos == string::npos) return -1;
    pos += label.length();
    while (pos < text.length() && text[pos] == ' ')  // skip any space(s) after the label
        pos++;
    string numStr;
    while (pos < text.length() && (isdigit(text[pos]) || text[pos] == '-'))
    {
        numStr += text[pos];
        pos++;
    }
    if (numStr.empty()) return -1;
    return atoi(numStr.c_str());
}

void slamDemo::qtNoticeHandler(const void *message)
{
    int index_, begin_, end_, feedback_, arrive_;
    const std_msgs::msg::dds_::String_ *seq = (const std_msgs::msg::dds_::String_ *)message;
    string str_, notice_;
    string fullText = seq->data();

    begin_ = fullText.find("index:", 0);
    end_ = fullText.find(";", begin_);
    str_ = fullText.substr(begin_ + 6, end_ - begin_ - 6);
    index_ = atoi(str_.c_str());

    begin_ = fullText.find("notice:", 0);
    end_ = fullText.find(";", begin_);
    notice_ = fullText.substr(begin_ + 7, end_ - begin_ - 7);

    if (index_ <= 10000)
    {
        begin_ = fullText.find("feedback:", 0);
        end_ = fullText.find(";", begin_);
        str_ = fullText.substr(begin_ + 9, end_ - begin_ - 9);
        feedback_ = atoi(str_.c_str());
        if (feedback_ == 0 || feedback_ == -1)
            cout << "\033[1;31m" << "Command execution failed with index = " << index_ << "." << "\033[0m";
        cout << notice_ << endl;
    }
    else if (index_ == 10001)
    {
        begin_ = fullText.find("arrive:", 0);
        end_ = fullText.find(";", begin_);
        str_ = fullText.substr(begin_ + 7, end_ - begin_ - 7);
        arrive_ = atoi(str_.c_str());
        cout << " I arrived node " << arrive_ << ". " << notice_ << endl;
        cout << "[DEBUG RAW] notice_=[" << notice_ << "] len=" << notice_.length() << endl;

        // ---- Watchdog logic ----
        bool isObstacleNotice = (notice_.find("obstacles ahead") != string::npos);

        std::lock_guard<std::mutex> lock(watchdogMutex);
        if (isObstacleNotice)
        {
            int goalNode = extractIntAfter(notice_, "Goal node name:");
            int edgeName = extractIntAfter(notice_, "Edge name:");

            if (edgeName != stuckEdgeName)
            {
                // New obstacle situation — start the clock fresh
                stuckEdgeName = edgeName;
                stuckGoalNode = goalNode;
                stuckSince = Clock::now();
                rerouteSentForThisStuck = false;
            }
            else if (!rerouteSentForThisStuck)
            {
                double secondsStuck = std::chrono::duration<double>(Clock::now() - stuckSince).count();
                if (secondsStuck >= OBSTACLE_TIMEOUT_SECONDS)
                {
                    cout << "\033[1;33m" << "[WATCHDOG] Stuck on edge " << stuckEdgeName
                         << " for " << (int)secondsStuck << "s. Rerouting directly to node "
                         << stuckGoalNode << " via single-node nav." << "\033[0m" << endl;
                    rerouteSentForThisStuck = true;
                    sendSingleNodeGoal(stuckGoalNode);
                }
            }
        }
        else
        {
            // Edge completed or notice changed — reset watchdog state
            stuckEdgeName = -1;
            stuckGoalNode = -1;
            rerouteSentForThisStuck = false;
        }
    }
}

void slamDemo::sendSingleNodeGoal(int target_node)
{
    if (target_node < 0) return;
    index++;
    unitree_interfaces::msg::dds_::QtCommand_ send_msg;
    send_msg.seq_().data() = "index:" + to_string(index) + ";";
    send_msg.command_() = 9; // single node navigation — bypasses edge routing
    send_msg.node_edge_name_().push_back(target_node);
    pubQtCommand->Write(send_msg);
}

void slamDemo::startMapping()
{
    index++;
    unitree_interfaces::msg::dds_::QtCommand_ send_msg;
    send_msg.seq_().data() = "index:" + to_string(index) + ";";
    send_msg.command_() = 3;
    send_msg.attribute_() = 1;
    pubQtCommand->Write(send_msg);
}

void slamDemo::endMapping()
{
    index++;
    unitree_interfaces::msg::dds_::QtCommand_ send_msg;
    send_msg.command_() = 4;
    send_msg.attribute_() = 0;
    send_msg.seq_().data() = "index:" + to_string(index) + ";";
    send_msg.floor_index_().push_back(0);
    send_msg.pcdmap_index_().push_back(0);
    pubQtCommand->Write(send_msg);
}

void slamDemo::startRelocation()
{
    index++;
    unitree_interfaces::msg::dds_::QtCommand_ send_msg;
    send_msg.seq_().data() = "index:" + to_string(index) + ";";
    send_msg.command_() = 6;
    send_msg.attribute_() = 1;
    pubQtCommand->Write(send_msg);
}

void slamDemo::initPose()
{
    index++;
    unitree_interfaces::msg::dds_::QtCommand_ send_msg;
    send_msg.seq_().data() = "index:" + to_string(index) + ";";
    send_msg.command_() = 7;
    send_msg.quaternion_x_() = 0;
    send_msg.quaternion_y_() = 0;
    send_msg.quaternion_z_() = 0;
    send_msg.quaternion_w_() = 1;
    send_msg.translation_x_() = 0;
    send_msg.translation_y_() = 0;
    send_msg.translation_z_() = 0;
    pubQtCommand->Write(send_msg);
}

void slamDemo::startNavigation()
{
    index++;
    unitree_interfaces::msg::dds_::QtCommand_ send_msg;
    send_msg.seq_().data() = "index:" + to_string(index) + ";";
    send_msg.command_() = 8;
    pubQtCommand->Write(send_msg);
}

void slamDemo::defaultNavigation()
{
    index++;
    unitree_interfaces::msg::dds_::QtCommand_ send_msg;
    send_msg.seq_().data() = "index:" + to_string(index) + ";";
    send_msg.command_() = 10;
    pubQtCommand->Write(send_msg);
}

void slamDemo::closeAllNode()
{
    index++;
    unitree_interfaces::msg::dds_::QtCommand_ send_msg;
    send_msg.seq_().data() = "index:" + to_string(index) + ";";
    send_msg.command_() = 99;
    pubQtCommand->Write(send_msg);
}

void slamDemo::addNodeAndEdge()
{
    nodeAttribute nodeTemp;
    float siny_cosp, cosy_cosp;

    siny_cosp = 2 * (currentOdom->pose().pose().orientation().w() * currentOdom->pose().pose().orientation().z() + currentOdom->pose().pose().orientation().x() * currentOdom->pose().pose().orientation().y());
    cosy_cosp = 1 - 2 * (currentOdom->pose().pose().orientation().y() * currentOdom->pose().pose().orientation().y() + currentOdom->pose().pose().orientation().z() * currentOdom->pose().pose().orientation().z());
    node_name++;

    nodeTemp.nodeX = currentOdom->pose().pose().position().x();
    nodeTemp.nodeY = currentOdom->pose().pose().position().y();
    nodeTemp.nodeZ = 0;
    nodeTemp.nodeYaw = std::atan2(siny_cosp, cosy_cosp);
    nodeTemp.nodeName = node_name;

    nodeAttributeList.push_back(nodeTemp);
    cout << "Add Node.   Name: " << node_name << "  X:" << nodeTemp.nodeX << "  Y:" << nodeTemp.nodeY << "  Z:" << nodeTemp.nodeZ << "  Yaw:" << nodeTemp.nodeYaw << endl;
    if (node_name >= 2)
        addEdge(node_name - 1, node_name - 1, node_name);
}

void slamDemo::addEdge(u_int16_t edge_name, u_int16_t start_node, u_int16_t end_node)
{
    edgeAttribute edgeTemp;
    edgeTemp.edgeName = edge_name;
    edgeTemp.edgeStart = start_node;
    edgeTemp.edgeEnd = end_node;
    edgeAttributeList.push_back(edgeTemp);
    cout << "Add Edge.   Name: " << edge_name << "  Start node:" << start_node << "  End node:" << end_node << endl;
    cout << "--------------------------------------------------------------------------" << endl;
}

void slamDemo::saveNodeAndEdge()
{
    unitree_interfaces::msg::dds_::QtNode_ nodeMsg;
    unitree_interfaces::msg::dds_::QtEdge_ edgeMsg;
    if (edgeAttributeList.size() == 0)
    {
        cout << "\033[1;31m" << "The number of edges in the topology graph is 0." << "\033[0m" << endl;
        return;
    }

    index++;
    nodeMsg.seq_().data() = "index:" + to_string(index) + ";";
    for (size_t i = 0; i < nodeAttributeList.size(); i++)
    {
        nodeMsg.node_().node_name_().push_back(nodeAttributeList[i].nodeName);
        nodeMsg.node_().node_position_x_().push_back(nodeAttributeList[i].nodeX);
        nodeMsg.node_().node_position_y_().push_back(nodeAttributeList[i].nodeY);
        nodeMsg.node_().node_position_z_().push_back(nodeAttributeList[i].nodeZ);
        nodeMsg.node_().node_yaw_().push_back(nodeAttributeList[i].nodeYaw);
        nodeMsg.node_().node_attribute_().push_back(0);
        nodeMsg.node_().undefined_().push_back(0);
        nodeMsg.node_().node_state_2_().push_back(0);
        nodeMsg.node_().node_state_3_().push_back(0);
    }
    pubQtNode->Write(nodeMsg);

    index++;
    edgeMsg.seq_().data() = "index:" + to_string(index) + ";";
    for (size_t i = 0; i < edgeAttributeList.size(); i++)
    {
        edgeMsg.edge_().edge_name_().push_back(edgeAttributeList[i].edgeName);
        edgeMsg.edge_().start_node_name_().push_back(edgeAttributeList[i].edgeStart);
        edgeMsg.edge_().end_node_name_().push_back(edgeAttributeList[i].edgeEnd);
        edgeMsg.edge_().edge_length_().push_back(0);
        edgeMsg.edge_().dog_speed_().push_back(0.5);
        edgeMsg.edge_().edge_state_2_().push_back(3);
        edgeMsg.edge_().dog_stats_().push_back(0);
        edgeMsg.edge_().dog_back_stats_().push_back(0);
        edgeMsg.edge_().edge_state_().push_back(0);
        edgeMsg.edge_().edge_state_1_().push_back(0);
        edgeMsg.edge_().edge_state_3_().push_back(0);
        edgeMsg.edge_().edge_state_4_().push_back(0);
    }
    pubQtEdge->Write(edgeMsg);
}

void slamDemo::addManualBypassEdge()
{
    int start_node, end_node, edge_name;
    cout << "Enter edge name (unique number): ";
    cin >> edge_name;
    cout << "Enter start node: ";
    cin >> start_node;
    cout << "Enter end node: ";
    cin >> end_node;
    cin.ignore(numeric_limits<streamsize>::max(), '\n');

    if (nodeAttributeList.size() == 0)
    {
        cout << "\033[1;31m" << "No node data available. Build and save a topology with 'x' and 'c' first." << "\033[0m" << endl;
        return;
    }

    unitree_interfaces::msg::dds_::QtNode_ nodeMsg;
    index++;
    nodeMsg.seq_().data() = "index:" + to_string(index) + ";";
    for (size_t i = 0; i < nodeAttributeList.size(); i++)
    {
        nodeMsg.node_().node_name_().push_back(nodeAttributeList[i].nodeName);
        nodeMsg.node_().node_position_x_().push_back(nodeAttributeList[i].nodeX);
        nodeMsg.node_().node_position_y_().push_back(nodeAttributeList[i].nodeY);
        nodeMsg.node_().node_position_z_().push_back(nodeAttributeList[i].nodeZ);
        nodeMsg.node_().node_yaw_().push_back(nodeAttributeList[i].nodeYaw);
        nodeMsg.node_().node_attribute_().push_back(0);
        nodeMsg.node_().undefined_().push_back(0);
        nodeMsg.node_().node_state_2_().push_back(0);
        nodeMsg.node_().node_state_3_().push_back(0);
    }
    pubQtNode->Write(nodeMsg);

    edgeAttribute edgeTemp;
    edgeTemp.edgeName = edge_name;
    edgeTemp.edgeStart = start_node;
    edgeTemp.edgeEnd = end_node;
    edgeAttributeList.push_back(edgeTemp);

    unitree_interfaces::msg::dds_::QtEdge_ edgeMsg;
    index++;
    edgeMsg.seq_().data() = "index:" + to_string(index) + ";";
    for (size_t i = 0; i < edgeAttributeList.size(); i++)
    {
        edgeMsg.edge_().edge_name_().push_back(edgeAttributeList[i].edgeName);
        edgeMsg.edge_().start_node_name_().push_back(edgeAttributeList[i].edgeStart);
        edgeMsg.edge_().end_node_name_().push_back(edgeAttributeList[i].edgeEnd);
        edgeMsg.edge_().edge_length_().push_back(0);
        edgeMsg.edge_().dog_speed_().push_back(0.5);
        edgeMsg.edge_().edge_state_2_().push_back(3);
        edgeMsg.edge_().dog_stats_().push_back(0);
        edgeMsg.edge_().dog_back_stats_().push_back(0);
        edgeMsg.edge_().edge_state_().push_back(0);
        edgeMsg.edge_().edge_state_1_().push_back(0);
        edgeMsg.edge_().edge_state_3_().push_back(0);
        edgeMsg.edge_().edge_state_4_().push_back(0);
    }
    pubQtEdge->Write(edgeMsg);

    cout << "Sent bypass edge " << edge_name << ": " << start_node << " -> " << end_node
         << " (index " << index << ") along with " << nodeAttributeList.size() << " nodes and "
         << edgeAttributeList.size() << " total edges." << endl;
}

void slamDemo::deleteAllNode()
{
    index++;
    unitree_interfaces::msg::dds_::QtCommand_ send_msg;
    send_msg.seq_().data() = "index:" + to_string(index) + ";";
    send_msg.command_() = 1;
    send_msg.attribute_() = 1;
    send_msg.node_edge_name_().push_back(999);
    pubQtCommand->Write(send_msg);
}

void slamDemo::deleteAllEdge()
{
    index++;
    unitree_interfaces::msg::dds_::QtCommand_ send_msg;
    send_msg.seq_().data() = "index:" + to_string(index) + ";";
    send_msg.command_() = 1;
    send_msg.attribute_() = 2;
    send_msg.node_edge_name_().push_back(999);
    pubQtCommand->Write(send_msg);
}

void slamDemo::pauseNavigation()
{
    index++;
    unitree_interfaces::msg::dds_::QtCommand_ send_msg;
    send_msg.seq_().data() = "index:" + to_string(index) + ";";
    send_msg.command_() = 13;
    pubQtCommand->Write(send_msg);
}

void slamDemo::recoverNavigation()
{
    index++;
    unitree_interfaces::msg::dds_::QtCommand_ send_msg;
    send_msg.seq_().data() = "index:" + to_string(index) + ";";
    send_msg.command_() = 14;
    pubQtCommand->Write(send_msg);
}

unsigned char slamDemo::keyDetection()
{
    termios tms_old, tms_new;
    tcgetattr(0, &tms_old);
    tms_new = tms_old;
    tms_new.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(0, TCSANOW, &tms_new);
    unsigned char ch = getchar();
    tcsetattr(0, TCSANOW, &tms_old);
    cout << "\033[1;32m" << "Key " << ch << " pressed." << "\033[0m" << endl;
    return ch;
}

void slamDemo::keyExecute()
{
    unsigned char currentKey;
    while (true)
    {
        currentKey = keyDetection();
        switch (currentKey)
        {
        case 'q': closeAllNode(); break;
        case 'w':
            deleteAllNode(); deleteAllEdge();
            nodeAttributeList.clear(); edgeAttributeList.clear();
            startMapping(); node_name = 0;
            break;
        case 'e': endMapping(); break;
        case 'a': startRelocation(); startNavigation(); initPose(); defaultNavigation(); break;
        case 's': pauseNavigation(); break;
        case 'd': recoverNavigation(); break;
        case 'z': deleteAllNode(); deleteAllEdge(); startRelocation(); startNavigation(); initPose(); break;
        case 'x': addNodeAndEdge(); break;
        case 'c': saveNodeAndEdge(); node_name = 0; break;
        case 'v':
            deleteAllNode(); deleteAllEdge();
            nodeAttributeList.clear(); edgeAttributeList.clear();
            node_name = 0;
            break;
        case 'b': addManualBypassEdge(); break;
        default: break;
        }
    }
}

int main(int argc, const char **argv)
{
    slamDemo slamTest(argv[1]);
    slamTest.keyExecute();
    return 0;
}
