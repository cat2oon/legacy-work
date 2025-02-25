from matrix import *
from context import *
from nets.leo import *
from ds.leo_data import *
from ds.leo_multi_data import *

np.set_printoptions(suppress=True)


"""
LEO 
    - config builder
    - loss unit test
"""

def set_gpu_memory_growth_mode(gpu_id=0):
    import tensorflow as tf
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(gpus[gpu_id], True)
    except RuntimeError as e:
        print(e)


def run_profile(profile_id, ctx, epoch=200, out_dir_path=None):
    # prepare
    mat = Matrix(ctx)
    net = Leo.create(ctx)
    mdp = MultiDataProvider(ctx, [profile_id])
    
    # run (TODO: 중간중간 평가셋 리포트)
    net.train(mdp, mat, num_epochs=epoch, is_leo=False)
    # net.eval(mdp)
    
    # TODO: 네트워크 저장하기
    
    # 오차 보고서 (TODO: refactoring)
    train_report = net.evaluate(mdp, use_cali_set=False, use_last_gen=False)
    tr_err = Matrix.compute_errors(train_report)
    valid_report = net.evaluate(mdp, use_cali_set=True, use_last_gen=False)
    vl_err = Matrix.compute_errors(valid_report)
    report = {'train':tr_err, 'valid':vl_err}
    
    # 보고서 저장
    if out_dir_path is not None:
        report_name = '{}.json'.format(profile_id)
        with open(os.path.join(out_dir_path, report_name), 'w') as f:
            json.dump(str(report), f)