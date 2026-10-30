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
#include <chrono>
#include <thread>
#include <mutex>
#include <atomic>
#include <set>
#include <stack>
#include <limits>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>

#define COMMANDTOPIC "rt/qt_command"
#define NOTICETOPIC "rt/qt_notice"
#define ODOMTOPIC "rt/lio_sam_ros2/mapping/re_location_odometry"
#define ADDNODETOPIC "rt/qt_add_node"
#define ADDEDGETOPIC "rt/qt_add_edge"

#define OBSTACLE_TIMEOUT_SECONDS 15
#define RELOCATION_SETTLE_MS 3000
#define PRIORITY_ARRIVAL_PAUSE_SECONDS 5

using namespace unitree::robot;
using namespace std;
using Clock = std::chrono::steady_clock;

static const string ORIGINAL_NODE_PATH = "/unitree/module/graph_pid_ws/src/graph_process/graph_map/Original_Graph/Original_Node.txt";
static const string ORIGINAL_EDGE_PATH = "/unitree/module/graph_pid_ws/src/graph_process/graph_map/Original_Graph/Original_Edge.txt";

// ----- console formatting helpers -----
static void hr(char c = '-', int width = 78) { cout << string(width, c) << "\n"; }
static void banner(const string &title)
{
    hr('=');
    int pad = (78 - (int)title.size()) / 2;
    if (pad < 0) pad = 0;
    cout << string(pad, ' ') << title << "\n";
    hr('=');
}
static void section(const string &title) { cout << "\n\033[1;36m" << title << "\033[0m\n"; }
static void keyLine(const string &key, const string &desc)
{
    cout << "  \033[1;33m" << left << setw(4) << key << "\033[0m " << desc << "\n";
}

struct nodeAttribute { u_int16_t nodeName; float nodeX, nodeY, nodeZ, nodeYaw; };
struct edgeAttribute { u_int16_t edgeName, edgeStart, edgeEnd; };

static int extractIntAfter(const string &text, const string &label)
{
    size_t pos = text.find(label);
    if (pos == string::npos) return -1;
    pos += label.length();
    while (pos < text.length() && text[pos] == ' ') pos++;
    string numStr;
    while (pos < text.length() && (isdigit(text[pos]) || text[pos] == '-')) { numStr += text[pos]; pos++; }
    if (numStr.empty()) return -1;
    return atoi(numStr.c_str());
}

// Trim + lowercase a token for command matching (so "S", " s ", "d" etc. all work)
static string normalizeToken(string s)
{
    size_t start = s.find_first_not_of(" \t\r\n");
    size_t end = s.find_last_not_of(" \t\r\n");
    if (start == string::npos) return "";
    s = s.substr(start, end - start + 1);
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
    return s;
}

class priorityNavDemo
{
private:
    int index = 0;
    const nav_msgs::msg::dds_::Odometry_ *currentOdom;
    u_int16_t node_name = 0;

    vector<nodeAttribute> nodeAttributeList;
    vector<edgeAttribute> edgeAttributeList;

    // ---- priority-goal navigation state (all guarded by navMutex) ----
    std::mutex navMutex;
    bool navigating = false;
    bool navPaused = false;                // true after 's' inside the goal loop, until 'd'
    int activeGoal = -1;
    std::stack<int> pausedGoals;
    int stuckEdgeName = -1;
    Clock::time_point stuckSince;
    bool rerouteSentForThisStuck = false;
    std::set<int> edgesRemovedThisSession;

    ChannelPublisherPtr<unitree_interfaces::msg::dds_::QtCommand_> pubQtCommand = ChannelPublisherPtr<unitree_interfaces::msg::dds_::QtCommand_>(new ChannelPublisher<unitree_interfaces::msg::dds_::QtCommand_>(COMMANDTOPIC));
    ChannelPublisherPtr<unitree_interfaces::msg::dds_::QtNode_> pubQtNode = ChannelPublisherPtr<unitree_interfaces::msg::dds_::QtNode_>(new ChannelPublisher<unitree_interfaces::msg::dds_::QtNode_>(ADDNODETOPIC));
    ChannelPublisherPtr<unitree_interfaces::msg::dds_::QtEdge_> pubQtEdge = ChannelPublisherPtr<unitree_interfaces::msg::dds_::QtEdge_>(new ChannelPublisher<unitree_interfaces::msg::dds_::QtEdge_>(ADDEDGETOPIC));
    ChannelSubscriberPtr<std_msgs::msg::dds_::String_> subQtNotice = ChannelSubscriberPtr<std_msgs::msg::dds_::String_>(new ChannelSubscriber<std_msgs::msg::dds_::String_>(NOTICETOPIC));
    ChannelSubscriberPtr<nav_msgs::msg::dds_::Odometry_> subOdommetry = ChannelSubscriberPtr<nav_msgs::msg::dds_::Odometry_>(new ChannelSubscriber<nav_msgs::msg::dds_::Odometry_>(ODOMTOPIC));

public:
    priorityNavDemo(const char *networkInterface);

    void qtNoticeHandler(const void *message);
    void odometryHandler(const void *message);

    void keyExecute();
    unsigned char keyDetection();
    void printMenu();

    void startMapping();
    void endMapping();
    void startRelocation();
    void initPose();
    void startNavigation();
    void sendGoalCommand(int target_node);
    void deleteSpecificEdge(int edgeName);
    void resendFullNodesAndEdges();
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

    void goalInputLoop();
};

void priorityNavDemo::printMenu()
{
    cout << "\n";
    banner("PRIORITY-GOAL NAVIGATION DEMO");
    cout << "   Same watchdog/self-heal as before, PLUS while navigating:\n"
         << "     - type a NODE NUMBER to interrupt with a priority goal\n"
         << "       (auto-resumes the original goal " << PRIORITY_ARRIVAL_PAUSE_SECONDS << "s after the interrupt goal is reached)\n"
         << "     - type 's' to PAUSE navigation, 'd' to RESUME it\n";
    hr();

    section("Mapping a new site");
    keyLine("w", "Start mapping");
    keyLine("e", "End mapping");
    keyLine("x", "Add a node at the robot's current position");
    keyLine("b", "Add a manual bypass edge");
    keyLine("c", "Push the current topology to the robot");

    section("Navigating");
    keyLine("a", "Start priority-aware goal-driven navigation");
    cout << "       -> auto-syncs the live graph from disk first.\n"
         << "       -> while navigating, the prompt accepts: a node number\n"
         << "          (interrupt), 's' (pause), or 'd' (resume).\n";
    keyLine("g", "Manually reload + re-push the live on-disk graph (no navigation)");

    section("Other controls (only active OUTSIDE the goal loop)");
    keyLine("z", "Raw relocate (no topology push)");
    keyLine("s", "Pause navigation");
    keyLine("d", "Recover / resume navigation");
    keyLine("v", "Clear all nodes/edges");
    keyLine("q", "Close");
    hr();
}

priorityNavDemo::priorityNavDemo(const char *networkInterface)
{
    ChannelFactory::Instance()->Init(0, networkInterface);
    pubQtCommand->InitChannel();
    pubQtNode->InitChannel();
    pubQtEdge->InitChannel();
    subQtNotice->InitChannel(std::bind(&priorityNavDemo::qtNoticeHandler, this, std::placeholders::_1), 10);
    subOdommetry->InitChannel(std::bind(&priorityNavDemo::odometryHandler, this, std::placeholders::_1), 1);
    printMenu();
}

void priorityNavDemo::odometryHandler(const void *message)
{
    currentOdom = (const nav_msgs::msg::dds_::Odometry_ *)message;
}

void priorityNavDemo::qtNoticeHandler(const void *message)
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
            cout << "\033[1;31m[ERROR] Command index " << index_ << " failed.\033[0m\n";
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

        std::lock_guard<std::mutex> lock(navMutex);
        if (!navigating) return;

        if (isCompletion)
        {
            stuckEdgeName = -1;
            rerouteSentForThisStuck = false;

            if (arrive_ == activeGoal)
            {
                if (!pausedGoals.empty())
                {
                    int justReached = activeGoal;
                    int resumeTarget = pausedGoals.top();
                    pausedGoals.pop();
                    cout << "\033[1;35m  [PRIORITY] Reached interrupt goal node " << justReached
                         << ". Pausing " << PRIORITY_ARRIVAL_PAUSE_SECONDS
                         << "s, then resuming node " << resumeTarget << "...\033[0m\n";

                    std::thread([this, resumeTarget]() {
                        std::this_thread::sleep_for(std::chrono::seconds(PRIORITY_ARRIVAL_PAUSE_SECONDS));
                        std::lock_guard<std::mutex> lock2(navMutex);
                        if (!navigating) return;
                        activeGoal = resumeTarget;
                        stuckEdgeName = -1;
                        rerouteSentForThisStuck = false;
                        sendGoalCommand(resumeTarget);
                        cout << "\033[1;35m  [PRIORITY] Resuming original goal: node " << resumeTarget << ".\033[0m\n";
                    }).detach();
                }
                else
                {
                    cout << "\033[1;32m  [GOAL] Final destination reached: node " << arrive_ << "!\033[0m\n";
                    if (!edgesRemovedThisSession.empty())
                    {
                        cout << "\033[1;32m  [SELF-HEAL] Restoring " << edgesRemovedThisSession.size()
                             << " edge(s) removed during this journey...\033[0m\n";
                        edgesRemovedThisSession.clear();
                        resendFullNodesAndEdges();
                    }
                    navigating = false;
                    navPaused = false;
                    activeGoal = -1;
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
                         << (int)secondsStuck << "s. Deleting it and resending current leg's goal...\033[0m\n";
                    rerouteSentForThisStuck = true;
                    edgesRemovedThisSession.insert(edgeName);
                    deleteSpecificEdge(edgeName);
                    if (activeGoal >= 0)
                        sendGoalCommand(activeGoal);
                }
            }
        }
    }
}

void priorityNavDemo::startMapping()
{
    index++;
    unitree_interfaces::msg::dds_::QtCommand_ m;
    m.seq_().data() = "index:" + to_string(index) + ";";
    m.command_() = 3; m.attribute_() = 1;
    pubQtCommand->Write(m);
}

void priorityNavDemo::endMapping()
{
    index++;
    unitree_interfaces::msg::dds_::QtCommand_ m;
    m.command_() = 4; m.attribute_() = 0;
    m.seq_().data() = "index:" + to_string(index) + ";";
    m.floor_index_().push_back(0);
    m.pcdmap_index_().push_back(0);
    pubQtCommand->Write(m);
}

void priorityNavDemo::startRelocation()
{
    index++;
    unitree_interfaces::msg::dds_::QtCommand_ m;
    m.seq_().data() = "index:" + to_string(index) + ";";
    m.command_() = 6; m.attribute_() = 1;
    pubQtCommand->Write(m);
}

void priorityNavDemo::initPose()
{
    index++;
    unitree_interfaces::msg::dds_::QtCommand_ m;
    m.seq_().data() = "index:" + to_string(index) + ";";
    m.command_() = 7;
    m.quaternion_x_() = 0; m.quaternion_y_() = 0; m.quaternion_z_() = 0; m.quaternion_w_() = 1;
    m.translation_x_() = 0; m.translation_y_() = 0; m.translation_z_() = 0;
    pubQtCommand->Write(m);
}

void priorityNavDemo::startNavigation()
{
    index++;
    unitree_interfaces::msg::dds_::QtCommand_ m;
    m.seq_().data() = "index:" + to_string(index) + ";";
    m.command_() = 8;
    pubQtCommand->Write(m);
}

void priorityNavDemo::sendGoalCommand(int target_node)
{
    index++;
    unitree_interfaces::msg::dds_::QtCommand_ m;
    m.seq_().data() = "index:" + to_string(index) + ";";
    m.command_() = 9;
    m.node_edge_name_().push_back(target_node);
    pubQtCommand->Write(m);
}

void priorityNavDemo::deleteSpecificEdge(int edgeName)
{
    index++;
    unitree_interfaces::msg::dds_::QtCommand_ m;
    m.seq_().data() = "index:" + to_string(index) + ";";
    m.command_() = 1; m.attribute_() = 2;
    m.node_edge_name_().push_back(edgeName);
    pubQtCommand->Write(m);
    cout << "\033[1;33m  [WATCHDOG] Sent delete for edge " << edgeName << ".\033[0m\n";
}

void priorityNavDemo::resendFullNodesAndEdges()
{
    if (nodeAttributeList.empty() || edgeAttributeList.empty())
    {
        cout << "\033[1;31m  [SELF-HEAL] Nothing to resend — local topology is empty.\033[0m\n";
        return;
    }
    unitree_interfaces::msg::dds_::QtNode_ nodeMsg;
    index++;
    nodeMsg.seq_().data() = "index:" + to_string(index) + ";";
    for (auto &n : nodeAttributeList)
    {
        nodeMsg.node_().node_name_().push_back(n.nodeName);
        nodeMsg.node_().node_position_x_().push_back(n.nodeX);
        nodeMsg.node_().node_position_y_().push_back(n.nodeY);
        nodeMsg.node_().node_position_z_().push_back(n.nodeZ);
        nodeMsg.node_().node_yaw_().push_back(n.nodeYaw);
        nodeMsg.node_().node_attribute_().push_back(0);
        nodeMsg.node_().undefined_().push_back(0);
        nodeMsg.node_().node_state_2_().push_back(0);
        nodeMsg.node_().node_state_3_().push_back(0);
    }
    pubQtNode->Write(nodeMsg);

    unitree_interfaces::msg::dds_::QtEdge_ edgeMsg;
    index++;
    edgeMsg.seq_().data() = "index:" + to_string(index) + ";";
    for (auto &e : edgeAttributeList)
    {
        edgeMsg.edge_().edge_name_().push_back(e.edgeName);
        edgeMsg.edge_().start_node_name_().push_back(e.edgeStart);
        edgeMsg.edge_().end_node_name_().push_back(e.edgeEnd);
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
         << edgeAttributeList.size() << " edges.\033[0m\n";
}

bool priorityNavDemo::loadLiveGraphFromDisk()
{
    ifstream nodeFile(ORIGINAL_NODE_PATH);
    ifstream edgeFile(ORIGINAL_EDGE_PATH);
    if (!nodeFile.is_open() || !edgeFile.is_open())
    {
        cout << "\033[1;31m  [GRAPH] Could not open live graph files.\033[0m\n";
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
        nodeAttribute n; int name;
        if (!(ss >> name >> n.nodeX >> n.nodeY >> n.nodeZ >> n.nodeYaw)) continue;
        n.nodeName = (u_int16_t)name;
        newNodes.push_back(n);
        if (n.nodeName > maxNode) maxNode = n.nodeName;
    }
    while (getline(edgeFile, line))
    {
        if (line.empty()) continue;
        stringstream ss(line);
        edgeAttribute e; int name, start, end;
        if (!(ss >> name >> start >> end)) continue;
        e.edgeName = (u_int16_t)name; e.edgeStart = (u_int16_t)start; e.edgeEnd = (u_int16_t)end;
        newEdges.push_back(e);
    }
    if (newNodes.empty() || newEdges.empty())
    {
        cout << "\033[1;31m  [GRAPH] Live graph parsed as empty — refusing to overwrite.\033[0m\n";
        return false;
    }
    nodeAttributeList = newNodes;
    edgeAttributeList = newEdges;
    node_name = maxNode;
    cout << "\n\033[1;32m  [GRAPH] Loaded live topology: " << nodeAttributeList.size()
         << " nodes, " << edgeAttributeList.size() << " edges.\033[0m\n";
    for (auto &e : edgeAttributeList)
        cout << "    edge " << left << setw(3) << e.edgeName << ": " << e.edgeStart << " -> " << e.edgeEnd << "\n";
    return true;
}

void priorityNavDemo::addNodeAndEdge()
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
    cout << "  Add Node.  Name:" << node_name << "  X:" << fixed << setprecision(3) << nodeTemp.nodeX
         << "  Y:" << nodeTemp.nodeY << "  Z:" << nodeTemp.nodeZ << "  Yaw:" << nodeTemp.nodeYaw << "\n";
    if (node_name >= 2) addEdge(node_name - 1, node_name - 1, node_name);
}

void priorityNavDemo::addEdge(u_int16_t edge_name, u_int16_t start_node, u_int16_t end_node)
{
    edgeAttribute e; e.edgeName = edge_name; e.edgeStart = start_node; e.edgeEnd = end_node;
    edgeAttributeList.push_back(e);
    cout << "  Add Edge.  Name:" << edge_name << "  Start:" << start_node << "  End:" << end_node << "\n";
}

void priorityNavDemo::saveNodeAndEdge()
{
    if (edgeAttributeList.empty())
    {
        cout << "\033[1;31m  No edges yet.\033[0m\n";
        return;
    }
    resendFullNodesAndEdges();
}

void priorityNavDemo::addManualBypassEdge()
{
    int start_node, end_node, edge_name;
    cout << "  Enter edge name: "; cin >> edge_name;
    cout << "  Enter start node: "; cin >> start_node;
    cout << "  Enter end node: "; cin >> end_node;
    cin.ignore(numeric_limits<streamsize>::max(), '\n');
    if (nodeAttributeList.empty())
    {
        cout << "\033[1;31m  No node data — press 'g' to load the live graph first.\033[0m\n";
        return;
    }
    edgeAttribute e; e.edgeName = edge_name; e.edgeStart = start_node; e.edgeEnd = end_node;
    edgeAttributeList.push_back(e);
    resendFullNodesAndEdges();
}

void priorityNavDemo::deleteAllNode()
{
    index++;
    unitree_interfaces::msg::dds_::QtCommand_ m;
    m.seq_().data() = "index:" + to_string(index) + ";";
    m.command_() = 1; m.attribute_() = 1;
    m.node_edge_name_().push_back(999);
    pubQtCommand->Write(m);
}

void priorityNavDemo::deleteAllEdge()
{
    index++;
    unitree_interfaces::msg::dds_::QtCommand_ m;
    m.seq_().data() = "index:" + to_string(index) + ";";
    m.command_() = 1; m.attribute_() = 2;
    m.node_edge_name_().push_back(999);
    pubQtCommand->Write(m);
}

void priorityNavDemo::pauseNavigation()
{
    index++;
    unitree_interfaces::msg::dds_::QtCommand_ m;
    m.seq_().data() = "index:" + to_string(index) + ";";
    m.command_() = 13;
    pubQtCommand->Write(m);
}

void priorityNavDemo::recoverNavigation()
{
    index++;
    unitree_interfaces::msg::dds_::QtCommand_ m;
    m.seq_().data() = "index:" + to_string(index) + ";";
    m.command_() = 14;
    pubQtCommand->Write(m);
}

void priorityNavDemo::closeAllNode()
{
    index++;
    unitree_interfaces::msg::dds_::QtCommand_ m;
    m.seq_().data() = "index:" + to_string(index) + ";";
    m.command_() = 99;
    pubQtCommand->Write(m);
}

unsigned char priorityNavDemo::keyDetection()
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

void priorityNavDemo::goalInputLoop()
{
    while (true)
    {
        bool isNavigatingNow, isPausedNow;
        int currentActive;
        {
            std::lock_guard<std::mutex> lock(navMutex);
            isNavigatingNow = navigating;
            isPausedNow = navPaused;
            currentActive = activeGoal;
        }

        if (isNavigatingNow)
            cout << "\n  Heading to node " << currentActive
                 << (isPausedNow ? " [PAUSED]" : "")
                 << " — enter a node to interrupt, 's' to pause, 'd' to resume: " << flush;
        else
            cout << "\n  Enter goal node (or -1 to return to main menu): " << flush;

        string raw;
        if (!(cin >> raw))
        {
            cin.clear();
            cin.ignore(numeric_limits<streamsize>::max(), '\n');
            continue;
        }
        cin.ignore(numeric_limits<streamsize>::max(), '\n');
        string token = normalizeToken(raw);

        std::lock_guard<std::mutex> lock(navMutex);

        // 's' / 'd' are only meaningful once navigation has actually started.
        if (navigating && token == "s")
        {
            pauseNavigation();
            navPaused = true;
            cout << "\033[1;33m  [NAV] Paused. Still heading to node " << activeGoal
                 << " once resumed.\033[0m\n";
            continue;
        }
        if (navigating && token == "d")
        {
            recoverNavigation();
            navPaused = false;
            cout << "\033[1;33m  [NAV] Resumed toward node " << activeGoal << ".\033[0m\n";
            continue;
        }

        // Otherwise, try to parse the token as a node number.
        bool isNumber = !token.empty() &&
            (isdigit((unsigned char)token[0]) || (token[0] == '-' && token.size() > 1));
        if (!isNumber)
        {
            cout << "\033[1;31m  [INFO] Unrecognized input '" << raw
                 << "'. Enter a node number, 's', or 'd'.\033[0m\n";
            continue;
        }
        int n = atoi(token.c_str());

        if (!navigating)
        {
            if (n < 0)
            {
                cout << "  Exiting goal mode.\n";
                return;
            }
            navigating = true;
            navPaused = false;
            activeGoal = n;
            stuckEdgeName = -1;
            rerouteSentForThisStuck = false;
            edgesRemovedThisSession.clear();
            while (!pausedGoals.empty()) pausedGoals.pop();
            sendGoalCommand(n);
            cout << "  [GOAL] Sent goal: node " << n << ". Waiting for arrival...\n";
        }
        else
        {
            if (n < 0)
            {
                cout << "\033[1;31m  [INFO] Navigation is in progress — 'pause' with 's' if you need to stop, "
                     << "or interrupt it with a new node number.\033[0m\n";
                continue;
            }
            if (n == activeGoal)
            {
                cout << "  [INFO] Already heading to node " << n << ".\n";
                continue;
            }
            cout << "\033[1;35m  [PRIORITY] Interrupting node " << activeGoal
                 << " -> new priority goal: node " << n
                 << ". Will resume node " << activeGoal << " after this.\033[0m\n";
            pausedGoals.push(activeGoal);
            activeGoal = n;
            navPaused = false;
            stuckEdgeName = -1;
            rerouteSentForThisStuck = false;
            sendGoalCommand(n);
        }
    }
}

void priorityNavDemo::keyExecute()
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
            cout << "\n  [MAP] Started mapping.\n";
            break;
        case 'e':
            endMapping();
            cout << "\n  [MAP] Ended mapping — PCD saved.\n";
            break;
        case 'a':
        {
            cout << "\n  [NAV] Syncing local topology from the live on-disk graph...\n";
            loadLiveGraphFromDisk();
            resendFullNodesAndEdges();
            startRelocation();
            startNavigation();
            initPose();
            cout << "\n  Waiting " << (RELOCATION_SETTLE_MS / 1000) << "s for relocation/navigation to settle...\n";
            std::this_thread::sleep_for(std::chrono::milliseconds(RELOCATION_SETTLE_MS));
            goalInputLoop();
            printMenu();
            break;
        }
        case 's': pauseNavigation(); cout << "\n  [NAV] Paused.\n"; break;
        case 'd': recoverNavigation(); cout << "\n  [NAV] Recovering.\n"; break;
        case 'z':
            deleteAllNode(); deleteAllEdge(); startRelocation(); startNavigation(); initPose();
            cout << "\n  [NAV] Raw relocate sent.\n";
            break;
        case 'x': addNodeAndEdge(); break;
        case 'c': saveNodeAndEdge(); node_name = 0; break;
        case 'v':
            deleteAllNode(); deleteAllEdge();
            nodeAttributeList.clear(); edgeAttributeList.clear();
            node_name = 0;
            cout << "\n  [MAP] Cleared all nodes/edges.\n";
            break;
        case 'b': addManualBypassEdge(); break;
        case 'g':
            cout << "\n  [GRAPH] Manual reload requested.\n";
            if (loadLiveGraphFromDisk()) resendFullNodesAndEdges();
            break;
        default: break;
        }
    }
}

int main(int argc, const char **argv)
{
    priorityNavDemo demo(argv[1]);
    demo.keyExecute();
    return 0;
}
