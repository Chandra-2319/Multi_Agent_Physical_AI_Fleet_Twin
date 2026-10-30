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
#include <thread>
#include <mutex>
#include <atomic>
#include <set>
#include <limits>
#include <fstream>
#include <sstream>
#include <iomanip>

#define COMMANDTOPIC "rt/qt_command"
#define NOTICETOPIC "rt/qt_notice"
#define ODOMTOPIC "rt/lio_sam_ros2/mapping/re_location_odometry"
#define ADDNODETOPIC "rt/qt_add_node"
#define ADDEDGETOPIC "rt/qt_add_edge"

#define OBSTACLE_TIMEOUT_SECONDS 15
#define RELOCATION_SETTLE_MS 3000

using namespace unitree::robot;
using namespace std;
using Clock = std::chrono::steady_clock;

// Live-on-disk graph, written/kept in sync by the backend (main_process/map_management).
// This is our source of truth for "what edges actually currently exist" — independent of
// whatever this program's local vectors happen to hold this session, and independent of
// site_manager.sh, which swaps THIS file's contents before the stack boots.
static const string ORIGINAL_NODE_PATH = "/unitree/module/graph_pid_ws/src/graph_process/graph_map/Original_Graph/Original_Node.txt";
static const string ORIGINAL_EDGE_PATH = "/unitree/module/graph_pid_ws/src/graph_process/graph_map/Original_Graph/Original_Edge.txt";

// ----- small console-formatting helpers -----
static void hr(char c = '-', int width = 78)
{
    cout << string(width, c) << "\n";
}
static void banner(const string &title)
{
    hr('=');
    int pad = (78 - (int)title.size()) / 2;
    if (pad < 0) pad = 0;
    cout << string(pad, ' ') << title << "\n";
    hr('=');
}
static void section(const string &title)
{
    cout << "\n\033[1;36m" << title << "\033[0m\n";
}
static void keyLine(const string &key, const string &desc)
{
    cout << "  \033[1;33m" << left << setw(4) << key << "\033[0m " << desc << "\n";
}

struct nodeAttribute
{
    u_int16_t nodeName;
    float nodeX, nodeY, nodeZ, nodeYaw;
};

struct edgeAttribute
{
    u_int16_t edgeName, edgeStart, edgeEnd;
};

static int extractIntAfter(const string &text, const string &label)
{
    size_t pos = text.find(label);
    if (pos == string::npos) return -1;
    pos += label.length();
    while (pos < text.length() && text[pos] == ' ')
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

class slamDemo
{
private:
    int index = 0;
    const nav_msgs::msg::dds_::Odometry_ *currentOdom;
    u_int16_t node_name = 0;

    vector<nodeAttribute> nodeAttributeList;
    vector<edgeAttribute> edgeAttributeList;

    std::atomic<int> currentGoalNode{-1};
    std::atomic<bool> goalReached{false};

    std::mutex watchdogMutex;
    int stuckEdgeName = -1;
    Clock::time_point stuckSince;
    bool rerouteSentForThisStuck = false;
    std::set<int> edgesRemovedForCurrentGoal;

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
    void sendGoalCommand(int target_node);
    void deleteSpecificEdge(int edgeName);
    void resendFullNodesAndEdges();
    void runGoalLoop();
    void addNodeAndEdge();
    void addEdge(u_int16_t edge_name, u_int16_t start_node, u_int16_t end_node);
    void saveNodeAndEdge();
    void addManualBypassEdge();
    void closeAllNode();
    void deleteAllNode();
    void deleteAllEdge();
    void pauseNavigation();
    void recoverNavigation();

    bool loadLiveGraphFromDisk();
    void printMenu();
};

void slamDemo::printMenu()
{
    cout << "\n";
    banner("UNITREE GO2 SLAM DEMO");
    cout << "   Manual Goal Navigation + Immediate Reroute Watchdog + Live-Graph Self-Heal\n";
    hr();

    section("Mapping a new site");
    keyLine("w", "Start mapping (walk the robot around the area)");
    keyLine("e", "End mapping (saves PCD map to disk)");
    keyLine("x", "Add a node at the robot's current position");
    keyLine("b", "Add a manual bypass edge between two existing nodes");
    keyLine("c", "Push the current topology to the robot");

    section("Navigating");
    keyLine("a", "Start goal-driven navigation");
    cout << "       -> automatically reloads the live graph from disk first,\n"
         << "          so the watchdog's self-heal always has real data to restore.\n";
    keyLine("g", "Manually reload + re-push the live on-disk graph (no navigation)");
    cout << "       -> use this after 'site_manager.sh load' if you ever want to\n"
         << "          sync without immediately starting a goal loop.\n";

    section("Other controls");
    keyLine("z", "Raw relocate (relocalize only, no topology push)");
    keyLine("s", "Pause navigation");
    keyLine("d", "Recover / resume navigation");
    keyLine("v", "Clear all nodes/edges (local + robot)");
    keyLine("q", "Close");

    section("Notes");
    cout << "  * 'a' asks for a goal node, drives there, then asks again.\n"
         << "    Enter -1 to return to this menu.\n";
    cout << "  * Watchdog: if the SAME edge stays obstructed for " << OBSTACLE_TIMEOUT_SECONDS
         << "s, it is deleted\n"
         << "    and the SAME goal is resent immediately (no retreat maneuver).\n";
    cout << "  * Every edge removed this way is restored TOGETHER the moment the\n"
         << "    goal is finally reached — self-heal works even on a fresh boot,\n"
         << "    since 'a' always loads the real graph from disk first.\n";
    cout << "  * Multi-site switching (save/load a full map+graph snapshot) is\n"
         << "    handled OUTSIDE this program — use ~/site_manager.sh on the shell.\n";
    hr();
}

slamDemo::slamDemo(const char *networkInterface)
{
    ChannelFactory::Instance()->Init(0, networkInterface);

    pubQtCommand->InitChannel();
    pubQtNode->InitChannel();
    pubQtEdge->InitChannel();

    subQtNotice->InitChannel(std::bind(&slamDemo::qtNoticeHandler, this, std::placeholders::_1), 10);
    subOdommetry->InitChannel(std::bind(&slamDemo::odometryHandler, this, std::placeholders::_1), 1);

    printMenu();
}

void slamDemo::odometryHandler(const void *message)
{
    currentOdom = (const nav_msgs::msg::dds_::Odometry_ *)message;
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
            cout << "\033[1;31m" << "[ERROR] Command index " << index_ << " failed." << "\033[0m\n";
        cout << "  " << notice_ << "\n";
        return;
    }
    else if (index_ == 10001)
    {
        begin_ = fullText.find("arrive:", 0);
        end_ = fullText.find(";", begin_);
        str_ = fullText.substr(begin_ + 7, end_ - begin_ - 7);
        arrive_ = atoi(str_.c_str());
        cout << "  [ARRIVE] node " << arrive_ << " — " << notice_ << "\n";

        bool isObstacle = (notice_.find("obstacles ahead") != string::npos);
        bool isCompletion = (notice_.find("Path completion") != string::npos);

        std::lock_guard<std::mutex> lock(watchdogMutex);

        if (isCompletion)
        {
            stuckEdgeName = -1;
            rerouteSentForThisStuck = false;

            if (arrive_ == currentGoalNode.load())
            {
                goalReached.store(true);
                if (!edgesRemovedForCurrentGoal.empty())
                {
                    cout << "\033[1;32m  [SELF-HEAL] Restoring " << edgesRemovedForCurrentGoal.size()
                         << " edge(s) removed during this journey...\033[0m\n";
                    edgesRemovedForCurrentGoal.clear();
                    resendFullNodesAndEdges();
                }
            }
        }
        else if (isObstacle)
        {
            int edgeName = extractIntAfter(notice_, "Edge name:");

            if (edgeName != stuckEdgeName)
            {
                stuckEdgeName = edgeName;
                stuckSince = Clock::now();
                rerouteSentForThisStuck = false;
            }
            else if (!rerouteSentForThisStuck)
            {
                double secondsStuck = std::chrono::duration<double>(Clock::now() - stuckSince).count();
                if (secondsStuck >= OBSTACLE_TIMEOUT_SECONDS)
                {
                    cout << "\033[1;33m  [WATCHDOG] Edge " << stuckEdgeName << " stuck for "
                         << (int)secondsStuck << "s. Deleting it and resending goal to replan...\033[0m\n";
                    rerouteSentForThisStuck = true;
                    edgesRemovedForCurrentGoal.insert(edgeName);
                    deleteSpecificEdge(edgeName);

                    int goal = currentGoalNode.load();
                    if (goal >= 0)
                        sendGoalCommand(goal);
                }
            }
        }
    }
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

void slamDemo::sendGoalCommand(int target_node)
{
    index++;
    unitree_interfaces::msg::dds_::QtCommand_ send_msg;
    send_msg.seq_().data() = "index:" + to_string(index) + ";";
    send_msg.command_() = 9;
    send_msg.node_edge_name_().push_back(target_node);
    pubQtCommand->Write(send_msg);
}

void slamDemo::deleteSpecificEdge(int edgeName)
{
    index++;
    unitree_interfaces::msg::dds_::QtCommand_ send_msg;
    send_msg.seq_().data() = "index:" + to_string(index) + ";";
    send_msg.command_() = 1;
    send_msg.attribute_() = 2;
    send_msg.node_edge_name_().push_back(edgeName);
    pubQtCommand->Write(send_msg);
    cout << "\033[1;33m  [WATCHDOG] Sent delete for edge " << edgeName << ".\033[0m\n";
}

void slamDemo::resendFullNodesAndEdges()
{
    if (nodeAttributeList.empty() || edgeAttributeList.empty())
    {
        cout << "\033[1;31m  [SELF-HEAL] Nothing to resend — local topology is empty. "
             << "Check the graph files on disk.\033[0m\n";
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

    cout << "\033[1;32m  [SELF-HEAL] Resent " << nodeAttributeList.size() << " nodes and "
         << edgeAttributeList.size() << " edges (full topology restored).\033[0m\n";
}

bool slamDemo::loadLiveGraphFromDisk()
{
    ifstream nodeFile(ORIGINAL_NODE_PATH);
    ifstream edgeFile(ORIGINAL_EDGE_PATH);

    if (!nodeFile.is_open() || !edgeFile.is_open())
    {
        cout << "\033[1;31m  [GRAPH] Could not open live graph files:\033[0m\n"
             << "    " << ORIGINAL_NODE_PATH << "\n"
             << "    " << ORIGINAL_EDGE_PATH << "\n";
        return false;
    }

    vector<nodeAttribute> newNodes;
    vector<edgeAttribute> newEdges;
    u_int16_t maxNode = 0;

    string line;
    while (getline(nodeFile, line))
    {
        if (line.empty()) continue;
        stringstream ss(line);
        nodeAttribute n;
        int name;
        if (!(ss >> name >> n.nodeX >> n.nodeY >> n.nodeZ >> n.nodeYaw))
            continue;
        n.nodeName = (u_int16_t)name;
        newNodes.push_back(n);
        if (n.nodeName > maxNode) maxNode = n.nodeName;
    }

    while (getline(edgeFile, line))
    {
        if (line.empty()) continue;
        stringstream ss(line);
        edgeAttribute e;
        int name, start, end;
        if (!(ss >> name >> start >> end))
            continue;
        e.edgeName = (u_int16_t)name;
        e.edgeStart = (u_int16_t)start;
        e.edgeEnd = (u_int16_t)end;
        newEdges.push_back(e);
    }

    if (newNodes.empty() || newEdges.empty())
    {
        cout << "\033[1;31m  [GRAPH] Live graph files parsed as empty — refusing to overwrite local topology.\033[0m\n";
        return false;
    }

    nodeAttributeList = newNodes;
    edgeAttributeList = newEdges;
    node_name = maxNode;

    cout << "\n\033[1;32m  [GRAPH] Loaded live topology from disk: " << nodeAttributeList.size()
         << " nodes, " << edgeAttributeList.size() << " edges.\033[0m\n";
    for (auto &e : edgeAttributeList)
        cout << "    edge " << left << setw(3) << e.edgeName << ": "
             << e.edgeStart << " -> " << e.edgeEnd << "\n";

    return true;
}

void slamDemo::runGoalLoop()
{
    cout << "\n  Waiting " << (RELOCATION_SETTLE_MS / 1000) << "s for relocation/navigation to settle...\n";
    std::this_thread::sleep_for(std::chrono::milliseconds(RELOCATION_SETTLE_MS));

    while (true)
    {
        int target;
        cout << "\n  Enter goal node (or -1 to return to main menu): " << flush;
        cin >> target;
        cin.ignore(numeric_limits<streamsize>::max(), '\n');

        if (target < 0)
        {
            cout << "  Exiting goal mode.\n";
            printMenu();
            return;
        }

        {
            std::lock_guard<std::mutex> lock(watchdogMutex);
            currentGoalNode.store(target);
            goalReached.store(false);
            stuckEdgeName = -1;
            rerouteSentForThisStuck = false;
            edgesRemovedForCurrentGoal.clear();
        }

        sendGoalCommand(target);
        cout << "  Goal sent: node " << target << ". Waiting for arrival...\n";

        while (!goalReached.load())
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }

        cout << "\033[1;32m  Reached node " << target << "!\033[0m\n";
    }
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
    cout << "  Add Node.  Name:" << node_name
         << "  X:" << fixed << setprecision(3) << nodeTemp.nodeX
         << "  Y:" << nodeTemp.nodeY << "  Z:" << nodeTemp.nodeZ
         << "  Yaw:" << nodeTemp.nodeYaw << "\n";
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
    cout << "  Add Edge.  Name:" << edge_name
         << "  Start:" << start_node << "  End:" << end_node << "\n";
}

void slamDemo::saveNodeAndEdge()
{
    if (edgeAttributeList.size() == 0)
    {
        cout << "\033[1;31m  No edges in the topology graph yet.\033[0m\n";
        return;
    }
    resendFullNodesAndEdges();
}

void slamDemo::addManualBypassEdge()
{
    int start_node, end_node, edge_name;
    cout << "  Enter edge name (unique number): ";
    cin >> edge_name;
    cout << "  Enter start node: ";
    cin >> start_node;
    cout << "  Enter end node: ";
    cin >> end_node;
    cin.ignore(numeric_limits<streamsize>::max(), '\n');

    if (nodeAttributeList.size() == 0)
    {
        cout << "\033[1;31m  No node data available. Press 'g' to load the live graph, or build one with 'x'/'c' first.\033[0m\n";
        return;
    }

    edgeAttribute edgeTemp;
    edgeTemp.edgeName = edge_name;
    edgeTemp.edgeStart = start_node;
    edgeTemp.edgeEnd = end_node;
    edgeAttributeList.push_back(edgeTemp);

    resendFullNodesAndEdges();
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

void slamDemo::closeAllNode()
{
    index++;
    unitree_interfaces::msg::dds_::QtCommand_ send_msg;
    send_msg.seq_().data() = "index:" + to_string(index) + ";";
    send_msg.command_() = 99;
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
        case 'q':
            cout << "\n  Closing.\n";
            closeAllNode();
            break;
        case 'w':
            deleteAllNode(); deleteAllEdge();
            nodeAttributeList.clear(); edgeAttributeList.clear();
            startMapping(); node_name = 0;
            cout << "\n  [MAP] Started mapping — walk the robot around the new area.\n";
            break;
        case 'e':
            endMapping();
            cout << "\n  [MAP] Ended mapping — PCD saved to disk.\n";
            break;
        case 'a':
        {
            cout << "\n  [NAV] Syncing local topology from the live on-disk graph before navigating...\n";
            loadLiveGraphFromDisk();
            resendFullNodesAndEdges();
            startRelocation();
            startNavigation();
            initPose();
            runGoalLoop();
            break;
        }
        case 's':
            pauseNavigation();
            cout << "\n  [NAV] Paused.\n";
            break;
        case 'd':
            recoverNavigation();
            cout << "\n  [NAV] Recovering/resuming.\n";
            break;
        case 'z':
            deleteAllNode(); deleteAllEdge(); startRelocation(); startNavigation(); initPose();
            cout << "\n  [NAV] Raw relocate sent (no topology pushed).\n";
            break;
        case 'x':
            addNodeAndEdge();
            break;
        case 'c':
            saveNodeAndEdge();
            node_name = 0;
            break;
        case 'v':
            deleteAllNode(); deleteAllEdge();
            nodeAttributeList.clear(); edgeAttributeList.clear();
            node_name = 0;
            cout << "\n  [MAP] Cleared all local + robot nodes/edges.\n";
            break;
        case 'b':
            addManualBypassEdge();
            break;
        case 'g':
            cout << "\n  [GRAPH] Manual reload requested.\n";
            if (loadLiveGraphFromDisk())
                resendFullNodesAndEdges();
            break;
        default:
            break;
        }
    }
}

int main(int argc, const char **argv)
{
    slamDemo slamTest(argv[1]);
    slamTest.keyExecute();
    return 0;
}
