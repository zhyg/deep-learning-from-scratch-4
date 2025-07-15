import numpy as np
import matplotlib.pyplot as plt
import os
import inspect


def argmax(xs):
    idxes = [i for i, x in enumerate(xs) if x == max(xs)]
    if len(idxes) == 1:
        return idxes[0]
    elif len(idxes) == 0:
        return np.random.choice(len(xs))

    selected = np.random.choice(idxes)
    return selected


def greedy_probs(Q, state, epsilon=0, action_size=4):
    qs = [Q[(state, action)] for action in range(action_size)]
    max_action = argmax(qs)  # OR np.argmax(qs)
    base_prob = epsilon / action_size
    action_probs = {action: base_prob for action in range(action_size)}  #{0: ε/4, 1: ε/4, 2: ε/4, 3: ε/4}
    action_probs[max_action] += (1 - epsilon)
    return action_probs


def plot_total_reward(reward_history):
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.plot(range(len(reward_history)), reward_history)
    plt.show()


def adaptive_plt_show(save_path=None, dpi=150, bbox_inches='tight', **kwargs):
    """
    自适应显示matplotlib图形：
    - 如果是GUI后端，直接显示图形窗口
    - 如果是非GUI后端，保存为图片文件
    
    Args:
        save_path (str, optional): 保存路径。如果为None，自动生成基于调用文件的路径
        dpi (int): 图像分辨率，默认150
        bbox_inches (str): 边界框设置，默认'tight'
        **kwargs: 传递给plt.savefig的其他参数
    """
    # 检测后端类型
    backend = plt.get_backend().lower()
    non_gui_backends = ['agg', 'svg', 'pdf', 'ps', 'cairo', 'gdk']
    
    if any(non_gui in backend for non_gui in non_gui_backends):
        # 非GUI后端，保存图片
        if save_path is None:
            # 自动生成保存路径
            caller_frame = inspect.currentframe().f_back
            caller_file = caller_frame.f_globals['__file__']
            caller_dir = os.path.dirname(caller_file)
            script_name = os.path.splitext(os.path.basename(caller_file))[0]
            save_path = os.path.join(caller_dir, f'{script_name}.png')
        
        plt.savefig(save_path, dpi=dpi, bbox_inches=bbox_inches, **kwargs)
        print(f"图像已保存到: {save_path}")
    else:
        # GUI后端，直接显示
        plt.show()


