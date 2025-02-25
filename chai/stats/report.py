from stats.profile_report import *

class Reports:
        
    @classmethod   
    def summary(cls, report_file_paths, eval_epoch):
        total_frames = 0
        accum_mean = 0.0
        accum_profile_mean = 0.0

        for rp in report_file_paths:
            report = ProfileReport.load_from(rp)
            
            if report.is_empty():
                continue
            
            e = report.evaluation_of(eval_epoch)
            accum_mean += e.total_mean * e.num_frames
            accum_profile_mean += e.total_mean
            total_frames += e.num_frames

        num_profiles = len(report_file_paths)
        print("average of profile", accum_profile_mean / num_profiles)
        print("average of total frames", accum_mean / total_frames)