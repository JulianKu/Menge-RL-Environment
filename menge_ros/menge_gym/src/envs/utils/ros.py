from geometry_msgs.msg import Pose
from visualization_msgs.msg import Marker
import subprocess
import threading
import queue
import psutil
from time import sleep
# import pylaunch as pl
from rospy import loginfo, logdebug, logerr
from signal import SIGTERM
from numpy import arccos
from typing import Tuple, Dict


def pose2array(pose: Pose) -> Tuple[float, float, float]:
    """
    extract 2D position and orientation from ROS geometry_msgs/Pose message

    :param pose: ROS geometry_msgs/Pose message

    :return: list [x, y, omega] (2D position x,y + orientation omega)
    """
    # only 2D poses (position x,y + rotation around z)
    x = pose.position.x
    y = pose.position.y
    # orientation quaternion
    omega = 2 * arccos(pose.orientation.w)

    return x, y, omega


def obstacle2array(obs_pose: Pose) -> Tuple[float, float]:
    """
    extract 2D position from ROS geometry_msgs/Pose message

    :param obs_pose: ROS geometry_msgs/Pose message

    :return: list [x, y] (2D position x,y)
    """
    # only 2D point objects (x,y - no orientation)
    x = obs_pose.position.x
    y = obs_pose.position.y
    return x, y


def marker2array(marker: Marker) -> Tuple[float, float, float, float]:
    """
    extract 2D position, orientation and radius from ROS geometry_msgs/Pose message

    :param marker: ROS visualization_msgs/Marker message

    :return: list [x, y, q_w, q_z, r] (2D position x,y + orientation omega + radius r)
    """
    # only 2D poses (position x,y + rotation around z)
    x = marker.pose.position.x
    y = marker.pose.position.y
    # orientation quaternion
    omega = 2 * arccos(marker.pose.orientation.w)

    # radius
    r = marker.scale.x / 2
    return x, y, omega, r


def start_roslaunch_file(pkg: str, launchfile: str, launch_cli_args: Dict[str, str] = None) -> subprocess.Popen:
    """
    start a ROS launchfile with arguments

    :param pkg: ROS package where launchfile is specified
    :param launchfile: name of the launch file
    :param launch_cli_args: dict of additional command line arguments {arg:value}

    :return: process of the launched file
    """

    # make sure launchfile is provided with extension
    if not str(launchfile).endswith('.launch'):
        launchfile += '.launch'

    launch_expr = ["roslaunch", pkg, launchfile]

    if launch_cli_args:
        # resolve cli arguments
        for key in launch_cli_args:
            launch_expr.append(str(key) + ':=' + str(launch_cli_args[key]))

    # launch file

    return subprocess.Popen(launch_expr)


# def launch(pkg: str, executable: str, launch_cli_args: Dict[str, str] = None) -> pl.PyRosLaunch:
#     """
#     start a ROS node with arguments
#
#     :param pkg: ROS package where node is specified
#     :param executable: name of the node's executable
#     :param launch_cli_args: dict of additional command line arguments {arg:value}
#
#     :return: process of the launched node
#     """
#     args = []
#     if launch_cli_args:
#         # resolve cli arguments
#         for key in launch_cli_args:
#             args.append("-%s %s" % (key, launch_cli_args[key]))
#
#     configs = [pl.Node(pkg, executable, executable,
#                        # output (optional)
#                        output="screen",
#                        # passing in args for executable (optional)
#                        args=" ".join(map(str, args)))
#                ]
#
#     p = pl.PyRosLaunch(configs)
#     p.start()
#
#     return p


def isProcessRunning(process_name: str) -> Tuple[bool, int]:
    """
    checks whether a process is already running

    :param process_name:
    :return:
    """
    core_running = False
    pid = None
    for proc in psutil.process_iter():
        try:
            if process_name in proc.name():
                pid = proc.pid
                core_running = True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return core_running, pid


def kill_child_processes(parent_pid, sig=SIGTERM):
    try:
        parent = psutil.Process(parent_pid)
        loginfo(parent)
    except psutil.NoSuchProcess:
        logerr("parent process not existing")
        return
    children = parent.children(recursive=True)
    loginfo(children)
    for process in children:
        loginfo("try to kill child: " + str(process))
        process.send_signal(sig)


def output_reader(proc: subprocess.Popen, out_queue: queue.Queue):
    for line in iter(proc.stdout.readline, b''):
        # loginfo(out_line.decode('utf-8'))
        out_queue.put(line.decode('utf-8'))


class ROSHandle:
    def __init__(self):
        self.processes = {}  # type: Dict[int, subprocess.Popen]
        self.threads = {}  # type: Dict[int, threading.Thread]
        self.queue = queue.Queue()

        # check if "roscore" running already
        self.core_running, self.core_PID = isProcessRunning("roscore")
        self.core_thread = None
        if not self.core_running:
            loginfo("No roscore running yet")
            loginfo("Start new roscore")
            self.core_process = subprocess.Popen(["roscore"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            # self.core_thread = threading.Thread(target=output_reader, args=(self.core_process, self.queue))
            # self.core_thread.daemon = True
            # self.core_thread.start()
            self.core_PID = self.core_process.pid
            self.core_running = True
            # sleep(2)
            self.log_output()

    def start_rosnode(self, pkg: str, executable: str, launch_cli_args: Dict[str, str] = None) -> int:
        """
        start a ROS node with arguments

        :param pkg: ROS package where node is specified
        :param executable: name of the node's executable
        :param launch_cli_args: dict of additional command line arguments {arg:value}
        """

        run_expr = ["rosrun", pkg, executable]

        if launch_cli_args:
            # resolve cli arguments
            for key in launch_cli_args:
                run_expr.append("-%s %s" % (key, launch_cli_args[key]))

        # run process
        proc = subprocess.Popen(run_expr, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        thr = threading.Thread(target=output_reader, args=(proc, self.queue))
        thr.daemon = True
        thr.start()
        pid = proc.pid
        # sleep(5)
        self.log_output()

        self.processes[pid] = proc
        self.threads[pid] = thr

        return pid

    def log_output(self):

        out_queue = self.queue
        print('--- Start processing queues')
        while True:
            try:
                print(out_queue.get_nowait())
            except queue.Empty:
                break
        print('--- Processing queues done')

    def terminate(self):
        """
        terminate all ros processes and the core itself
        """
        loginfo("Trying to kill all launched processes first")
        for pid, process in self.processes.items():
            process.terminate()
            process.wait()
            self.threads[pid].join()
        if self.core_thread:
            loginfo("Trying to kill further child pids of roscore pid: " + str(self.core_PID))
            kill_child_processes(self.core_PID)
            self.core_process.terminate()
            self.core_process.wait()  # important to prevent from zombie process
            # self.core_thread.join()
        self.log_output()

    def terminateOne(self, pid):
        """
        terminate one ros process by its pid

        :param pid: process id for process to kill
        """
        loginfo("trying to kill process with pid: %s" % pid)
        proc = self.processes[pid]
        proc.terminate()
        proc.wait()
        self.threads[pid].join()
        self.log_output()
