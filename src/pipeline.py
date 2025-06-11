import os
from .steps.key_points_step.step_pipeline import Key_Points_step
from .steps.depth_step.step_pipeline import Depth_Map_step
from .steps.segmentation_step.step_pipeline import Segment_step
from .steps.get_separeted_walls.step_pipeline import Get_Separeted_walls_step
from .steps.make_3d_plan_step.step_pipeline import Make_3d_Plan_step
from .steps.make_2d_plan_step.step_pipeline import Make_2d_Plan_step
from .steps.add_door_window_step.step_pipeline import Add_door_window_step

from .steps.union_plan_step.union_plan import Union_Plan_step

class Pipeline:
    def __init__(self,flat_path):
        self.flat_path = flat_path

        self.results = {}
        self.steps = (Key_Points_step,
                      Depth_Map_step,
                      Segment_step,
                      Get_Separeted_walls_step,
                      Make_3d_Plan_step,
                      Make_2d_Plan_step,
                      Add_door_window_step)
        
    def run(self, viz = False):
        for room in os.listdir(os.path.join(self.flat_path, 'by_rooms')):
            for step in self.steps:
                if self.results is not None:
                    results = step(self.flat_path, room, self.results).run(viz)
                else:
                    results = step(self.flat_path, room).run(viz)
            self.results = {room: results}
        Union_Plan_step(self.flat_path).run()
        