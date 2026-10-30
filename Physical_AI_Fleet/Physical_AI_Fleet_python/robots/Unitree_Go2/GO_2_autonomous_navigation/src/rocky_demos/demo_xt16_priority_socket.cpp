#include "unitree/common/dds/dds_easy_model.hpp"
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

#include <unitree/ros2_idl/QtCommand_.hpp>
#include <unitree/ros2_idl/String_.hpp>
#include <unitree/ros2_idl/Odometry_.hpp>
#include <unitree/ros2_idl/QtNode_.hpp>
#include <unitree/ros2_idl/QtEdge_.hpp>

#include <cstdio>
#include <cmath>
#include <chrono>
#include <thread>
#include <mutex>
#include <atomic>
#include <set>
#include <stack>
#include <vector>
#include <fstream>
#include <sstream>
#include <cstring>
#include <algorithm>

#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>

#define COMMANDTOPIC "rt/qt_command"
#define NOTICETOPIC "rt/qt_notice"
#define ODOMTOPIC "rt/lio_sam_ros2/mapping/re_location_odometry"
#define ADDNODETOPIC "rt/qt_add_node"
#define ADDEDGETOPIC "rt/qt_add_edge"

#define OBSTACLE_TIMEOUT_SECONDS 15
#define RELOCATION_SETTLE_MS 3000
#define PRIORITY_ARRIVAL_PAUSE_SECONDS 5
#define SOCKET_PORT 9091

using namespace unitree::robot;
using namespace std;
using Clock = std::chrono::steady_clock;

static const string ORIGINAL_NODE_PATH = "/unitree/module/graph_pid_ws/src/graph_process/graph_map/Original_Graph/Original_Node.txt";
static const string ORIGINAL_EDGE_PATH = "/unitree/module/graph_pid_ws/src/graph_process/graph_map/Original_Graph/Original_Edge.txt";

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

class priorityNavSocket
{
private:
    int index = 0;
    const nav_msgs::msg::dds_::Odometry_ *currentOdom;
    u_int16_t node_name = 0;

    vector<nodeAttribute> nodeAttributeList;
    vector<edgeAttribute> edgeAttributeList;

    std::mutex navMutex;
    bool navigating = false;
    int activeGoal = -1;
    int homeNode = -1;          // -1 = no auto-return requested for this chain
    bool returningHome = false;
    std::stack<int> pausedGoals;
    int stuckEdgeName = -1;
    Clock::time_point stuckSince;
    bool rerouteSentForThisStuck = false;
    std::set<int> edgesRemovedThisSession;

    // Connected bridge socket(s)
    std::mutex clientMutex;
    vector<int> clientFds;

    ChannelPublisherPtr<unitree_interfaces::msg::dds_::QtCommand_> pubQtCommand = ChannelPublisherPtr<unitree_interfaces::msg::dds_::QtCommand_>(new ChannelPublisher<unitree_interfaces::msg::dds_::QtCommand_>(COMMANDTOPIC));
    ChannelPublisherPtr<unitree_interfaces::msg::dds_::QtNode_> pubQtNode = ChannelPublisherPtr<unitree_interfaces::msg::dds_::QtNode_>(new ChannelPublisher<unitree_interfaces::msg::dds_::QtNode_>(ADDNODETOPIC));
    ChannelPublisherPtr<unitree_interfaces::msg::dds_::QtEdge_> pubQtEdge = ChannelPublisherPtr<unitree_interfaces::msg::dds_::QtEdge_>(new ChannelPublisher<unitree_interfaces::msg::dds_::QtEdge_>(ADDEDGETOPIC));
    ChannelSubscriberPtr<std_msgs::msg::dds_::String_> subQtNotice = ChannelSubscriberPtr<std_msgs::msg::dds_::String_>(new ChannelSubscriber<std_msgs::msg::dds_::String_>(NOTICETOPIC));
    ChannelSubscriberPtr<nav_msgs::msg::dds_::Odometry_> subOdommetry = ChannelSubscriberPtr<nav_msgs::msg::dds_::Odometry_>(new ChannelSubscriber<nav_msgs::msg::dds_::Odometry_>(ODOMTOPIC));

public:
    priorityNavSocket(const char *networkInterface);

    void qtNoticeHandler(const void *message);
    void odometryHandler(const void *message);

    void sendGoalCommand(int target_node);
    void deleteSpecificEdge(int edgeName);
    void resendFullNodesAndEdges();
    bool loadLiveGraphFromDisk();

    void broadcastEvent(const string &line);
    void runSocketServer();
    void handleClient(int fd);
    void handleCommandLine(const string &line);
    void startNavigationBackend(); // relocation + settle, run once at startup
};

priorityNavSocket::priorityNavSocket(const char *networkInterface)
{
    ChannelFactory::Instance()->Init(0, networkInterface);
    pubQtCommand->InitChannel();
    pubQtNode->InitChannel();
    pubQtEdge->InitChannel();
    subQtNotice->InitChannel(std::bind(&priorityNavSocket::qtNoticeHandler, this, std::placeholders::_1), 10);
    subOdommetry->InitChannel(std::bind(&priorityNavSocket::odometryHandler, this, std::placeholders::_1), 1);

    cout << "[NAV-SOCKET] Started. Listening on port " << SOCKET_PORT << endl;
}

void priorityNavSocket::odometryHandler(const void *message)
{
    currentOdom = (const nav_msgs::msg::dds_::Odometry_ *)message;
}

void priorityNavSocket::broadcastEvent(const string &line)
{
    cout << line << endl;
    std::lock_guard<std::mutex> lock(clientMutex);
    string withNewline = line + "\n";
    for (size_t i = 0; i < clientFds.size(); )
    {
        ssize_t sent = send(clientFds[i], withNewline.c_str(), withNewline.size(), MSG_NOSIGNAL);
        if (sent < 0)
        {
            close(clientFds[i]);
            clientFds.erase(clientFds.begin() + i);
        }
        else i++;
    }
}

void priorityNavSocket::qtNoticeHandler(const void *message)
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
            cout << "[ERROR] Command index " << index_ << " failed. " << notice_ << endl;
        return;
    }
    else if (index_ == 10001)
    {
        begin_ = fullText.find("arrive:", 0);
        end_ = fullText.find(";", begin_);
        str_ = fullText.substr(begin_ + 7, end_ - begin_ - 7);
        arrive_ = atoi(str_.c_str());

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
                    broadcastEvent("EVT INTERRUPT_REACHED " + to_string(justReached) + " " + to_string(resumeTarget));

                    std::thread([this, resumeTarget]() {
                        std::this_thread::sleep_for(std::chrono::seconds(PRIORITY_ARRIVAL_PAUSE_SECONDS));
                        std::lock_guard<std::mutex> lock2(navMutex);
                        if (!navigating) return;
                        activeGoal = resumeTarget;
                        stuckEdgeName = -1;
                        rerouteSentForThisStuck = false;
                        sendGoalCommand(resumeTarget);
                        broadcastEvent("EVT RESUMING " + to_string(resumeTarget));
                    }).detach();
                }
                else if (returningHome)
                {
                    broadcastEvent("EVT HOME_REACHED " + to_string(arrive_));
                    navigating = false;
                    returningHome = false;
                    activeGoal = -1;
                    homeNode = -1;
                }
                else
                {
                    broadcastEvent("EVT FINAL_REACHED " + to_string(arrive_));
                    if (!edgesRemovedThisSession.empty())
                    {
                        edgesRemovedThisSession.clear();
                        resendFullNodesAndEdges();
                        broadcastEvent("EVT SELF_HEAL " + to_string(edgeAttributeList.size()));
                    }

                    if (homeNode >= 0 && homeNode != arrive_)
                    {
                        returningHome = true;
                        activeGoal = homeNode;
                        stuckEdgeName = -1;
                        rerouteSentForThisStuck = false;
                        sendGoalCommand(homeNode);
                        broadcastEvent("EVT RETURNING_HOME " + to_string(homeNode));
                    }
                    else
                    {
                        navigating = false;
                        activeGoal = -1;
                        homeNode = -1;
                    }
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
                    rerouteSentForThisStuck = true;
                    edgesRemovedThisSession.insert(edgeName);
                    deleteSpecificEdge(edgeName);
                    broadcastEvent("EVT WATCHDOG_STUCK " + to_string(edgeName));
                    if (activeGoal >= 0)
                        sendGoalCommand(activeGoal);
                }
            }
        }
    }
}

void priorityNavSocket::sendGoalCommand(int target_node)
{
    index++;
    unitree_interfaces::msg::dds_::QtCommand_ m;
    m.seq_().data() = "index:" + to_string(index) + ";";
    m.command_() = 9;
    m.node_edge_name_().push_back(target_node);
    pubQtCommand->Write(m);
}

void priorityNavSocket::deleteSpecificEdge(int edgeName)
{
    index++;
    unitree_interfaces::msg::dds_::QtCommand_ m;
    m.seq_().data() = "index:" + to_string(index) + ";";
    m.command_() = 1; m.attribute_() = 2;
    m.node_edge_name_().push_back(edgeName);
    pubQtCommand->Write(m);
}

void priorityNavSocket::resendFullNodesAndEdges()
{
    if (nodeAttributeList.empty() || edgeAttributeList.empty()) return;

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
}

bool priorityNavSocket::loadLiveGraphFromDisk()
{
    ifstream nodeFile(ORIGINAL_NODE_PATH);
    ifstream edgeFile(ORIGINAL_EDGE_PATH);
    if (!nodeFile.is_open() || !edgeFile.is_open())
    {
        cout << "[GRAPH] Could not open live graph files." << endl;
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
        cout << "[GRAPH] Live graph parsed as empty — refusing to overwrite." << endl;
        return false;
    }
    nodeAttributeList = newNodes;
    edgeAttributeList = newEdges;
    node_name = maxNode;
    cout << "[GRAPH] Loaded live topology: " << nodeAttributeList.size()
         << " nodes, " << edgeAttributeList.size() << " edges." << endl;
    return true;
}

void priorityNavSocket::startNavigationBackend()
{
    loadLiveGraphFromDisk();
    resendFullNodesAndEdges();

    // startRelocation (6), startNavigation (8), initPose (7)
    index++;
    { unitree_interfaces::msg::dds_::QtCommand_ m; m.seq_().data() = "index:" + to_string(index) + ";"; m.command_() = 6; m.attribute_() = 1; pubQtCommand->Write(m); }
    index++;
    { unitree_interfaces::msg::dds_::QtCommand_ m; m.seq_().data() = "index:" + to_string(index) + ";"; m.command_() = 8; pubQtCommand->Write(m); }
    index++;
    { unitree_interfaces::msg::dds_::QtCommand_ m; m.seq_().data() = "index:" + to_string(index) + ";"; m.command_() = 7;
      m.quaternion_x_() = 0; m.quaternion_y_() = 0; m.quaternion_z_() = 0; m.quaternion_w_() = 1;
      m.translation_x_() = 0; m.translation_y_() = 0; m.translation_z_() = 0; pubQtCommand->Write(m); }

    cout << "[NAV-SOCKET] Waiting " << (RELOCATION_SETTLE_MS / 1000) << "s to settle..." << endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(RELOCATION_SETTLE_MS));
    cout << "[NAV-SOCKET] Ready for commands." << endl;
}

// Parses: "GOAL <node>" or "GOAL <node> HOME <homeNode>" or "PAUSE" or "RESUME"
void priorityNavSocket::handleCommandLine(const string &lineIn)
{
    string line = lineIn;
    while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) line.pop_back();
    if (line.empty()) return;

    stringstream ss(line);
    string cmd; ss >> cmd;
    std::transform(cmd.begin(), cmd.end(), cmd.begin(), ::toupper);

    std::lock_guard<std::mutex> lock(navMutex);

    if (cmd == "PAUSE")
    {
        index++;
        unitree_interfaces::msg::dds_::QtCommand_ m; m.seq_().data() = "index:" + to_string(index) + ";"; m.command_() = 13; pubQtCommand->Write(m);
        return;
    }
    if (cmd == "RESUME")
    {
        index++;
        unitree_interfaces::msg::dds_::QtCommand_ m; m.seq_().data() = "index:" + to_string(index) + ";"; m.command_() = 14; pubQtCommand->Write(m);
        return;
    }
    if (cmd == "GOAL")
    {
        int n; ss >> n;
        int homeArg = -1;
        string maybeHomeKw; ss >> maybeHomeKw;
        if (!maybeHomeKw.empty())
        {
            std::transform(maybeHomeKw.begin(), maybeHomeKw.end(), maybeHomeKw.begin(), ::toupper);
            if (maybeHomeKw == "HOME") ss >> homeArg;
        }

        if (!navigating)
        {
            navigating = true;
            returningHome = false;
            activeGoal = n;
            homeNode = homeArg;
            stuckEdgeName = -1;
            rerouteSentForThisStuck = false;
            edgesRemovedThisSession.clear();
            while (!pausedGoals.empty()) pausedGoals.pop();
            sendGoalCommand(n);
            broadcastEvent("EVT GOAL_SENT " + to_string(n));
        }
        else
        {
            if (n == activeGoal) return;
            // Priority interrupt — same behavior regardless of whether the
            // current activeGoal is itself an interrupt, a normal goal, or
            // a home-return leg.
            broadcastEvent("EVT PRIORITY_INTERRUPT " + to_string(activeGoal) + " " + to_string(n));
            pausedGoals.push(activeGoal);
            activeGoal = n;
            stuckEdgeName = -1;
            rerouteSentForThisStuck = false;
            sendGoalCommand(n);
        }
    }
}

void priorityNavSocket::handleClient(int fd)
{
    {
        std::lock_guard<std::mutex> lock(clientMutex);
        clientFds.push_back(fd);
    }
    cout << "[NAV-SOCKET] Bridge client connected." << endl;

    char buf[1024];
    string accum;
    while (true)
    {
        ssize_t n = recv(fd, buf, sizeof(buf), 0);
        if (n <= 0) break;
        accum.append(buf, n);
        size_t pos;
        while ((pos = accum.find('\n')) != string::npos)
        {
            string line = accum.substr(0, pos);
            accum.erase(0, pos + 1);
            handleCommandLine(line);
        }
    }

    {
        std::lock_guard<std::mutex> lock(clientMutex);
        clientFds.erase(std::remove(clientFds.begin(), clientFds.end(), fd), clientFds.end());
    }
    close(fd);
    cout << "[NAV-SOCKET] Bridge client disconnected." << endl;
}

void priorityNavSocket::runSocketServer()
{
    int serverFd = socket(AF_INET, SOCK_STREAM, 0);
    int opt = 1;
    setsockopt(serverFd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(SOCKET_PORT);

    if (bind(serverFd, (sockaddr*)&addr, sizeof(addr)) < 0)
    {
        perror("[NAV-SOCKET] bind failed");
        return;
    }
    listen(serverFd, 4);
    cout << "[NAV-SOCKET] Listening on 0.0.0.0:" << SOCKET_PORT << endl;

    while (true)
    {
        sockaddr_in clientAddr{};
        socklen_t clientLen = sizeof(clientAddr);
        int clientFd = accept(serverFd, (sockaddr*)&clientAddr, &clientLen);
        if (clientFd < 0) continue;
        std::thread(&priorityNavSocket::handleClient, this, clientFd).detach();
    }
}

int main(int argc, const char **argv)
{
    if (argc < 2)
    {
        cout << "Usage: " << argv[0] << " networkInterface" << endl;
        return -1;
    }
    priorityNavSocket nav(argv[1]);
    nav.startNavigationBackend();
    nav.runSocketServer(); // blocks forever
    return 0;
}
