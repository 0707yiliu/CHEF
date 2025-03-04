import time

import pyschunk.tools.mylogger
from bkstools.bks_lib.bks_base import keep_communication_alive_input
from bkstools.bks_lib.bks_module import BKSModule, HandleWarningPrintOnly  # @UnusedImport
from bkstools.bks_lib.debug import Print, Var, ApplicationError, g_logmethod  # @UnusedImport

logger = pyschunk.tools.mylogger.getLogger( "BKSTools.demo.demo_grip_workpiece_with_position" )
pyschunk.tools.mylogger.setupLogging()
g_logmethod = logger.info
from pyschunk.generated.generated_enums import eCmdCode
from bkstools.bks_lib import bks_options

import yaml

class schunk_gripper_grasp:
    def __init__(self):

        self.port = '/dev/ttyUSB0'
        self.bks = BKSModule(self.port,
                        sleep_time=None,
                        # handle_warning=HandleWarningPrintOnly,
                        debug=False,
                        repeater_timeout=3.0,
                        repeater_nb_tries=5
                        )
        print("Make schunk ready...")
        self.bks.MakeReady()
        self.move_rel_velocity_ums = int(self.bks.max_vel * 1000.0)
        self.grip_direction = BKSModule.grip_from_outside
        # print(self.bks.max_vel, self.grip_direction)

        self._init_gripper()

    def _init_gripper(self):
        print('open finger for initialization.')
        self.bks.move_to_absolute_position(100, 50000)
        # time.sleep(3)
        self.bks.MakeReady()

    def impedance_grasp_with_MagTac(self):
        self._init_gripper()
        while True:
            # print('hi there')
            time.sleep(0.01)
            self.bks.move_to_relative_position(int(0.5 * 1000.0), self.move_rel_velocity_ums)

schunktest = schunk_gripper_grasp()
schunktest.impedance_grasp_with_MagTac()


