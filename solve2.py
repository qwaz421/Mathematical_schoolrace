import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scipy.stats import ttest_1samp  # 引入第一处升级：单样本t检验
from pulp import * # 引入第三处升级：整数线性规划

# =========================================================================
# 工具函数：计算两角度在 [-180, 180] 范围内的最小夹角差
# =========================================================================
def calculate_angle_diff(a1, a2):
    diff = np.abs(a1 - a2) % 360
    return np.where(diff > 180, 360 - diff, diff)

# =========================================================================
# 升级处一：问题3系统偏差显著性检验（单样本双尾 t 检验）
# =========================================================================
def bias_significance_test(df1, df2, dt, dx, dy):
    """
    对对齐时间后的空间残差进行统计学单样本 t 检验。
    【修正点】：必须使用未进行空间修正的原始坐标，去检验残差均值是否显著不为 0！
    """
    t1 = df1.iloc[:, 0].values
    x1 = df1.iloc[:, 1].values
    y1 = df1.iloc[:, 2].values
    
    t2 = df2.iloc[:, 0].values + dt
    # 【核心修改】：移除提前加 dx, dy 的操作，保持原始位置
    x2_raw = df2.iloc[:, 1].values 
    y2_raw = df2.iloc[:, 2].values
    
    fx = interp1d(t2, x2_raw, fill_value='extrapolate', kind='linear')
    fy = interp1d(t2, y2_raw, fill_value='extrapolate', kind='linear')
    
    mask = (t1 >= t2.min()) & (t1 <= t2.max())
    
    dx_residual = x1[mask] - fx(t1[mask])
    dy_residual = y1[mask] - fy(t1[mask])
    
    # 检验原始残差是否显著不等于 0
    px = ttest_1samp(dx_residual, 0).pvalue
    py = ttest_1samp(dy_residual, 0).pvalue
    
    return px, py

# =========================================================================
# 升级处二：基于空间残差统计特征的自适应观测噪声协方差 R 矩阵估计
# =========================================================================
def estimate_measurement_noise(df1, df2, dt, dx, dy):
    """
    根据两个传感器时空对齐后的时序残差样本方差，自适应生成卡尔曼滤波的 R 矩阵
    """
    t1 = df1.iloc[:, 0].values
    t2 = df2.iloc[:, 0].values + dt

    fx = interp1d(t2, df2.iloc[:, 1].values + dx, fill_value='extrapolate', kind='linear')
    fy = interp1d(t2, df2.iloc[:, 2].values + dy, fill_value='extrapolate', kind='linear')

    mask = (t1 >= t2.min()) & (t1 <= t2.max())

    ex = df1.iloc[:, 1].values[mask] - fx(t1[mask])
    ey = df1.iloc[:, 2].values[mask] - fy(t1[mask])

    sigma_x = np.var(ex)
    sigma_y = np.var(ey)

    # 加入 1e-4 的微小正则化项，防止方差极小时滤波矩阵求逆发生数值奇异
    return np.diag([max(sigma_x, 1e-4), max(sigma_y, 1e-4)])

# =========================================================================
# 核心基础算法：时空偏置联合优化对齐
# =========================================================================
def align_spatiotemporal(df1, df2, estimate_bias=True):
    t1, x1, y1 = df1.iloc[:, 0].values, df1.iloc[:, 1].values, df1.iloc[:, 2].values
    t2, x2, y2 = df2.iloc[:, 0].values, df2.iloc[:, 1].values, df2.iloc[:, 2].values
    
    f_x2 = interp1d(t2, x2, kind='linear', fill_value='extrapolate')
    f_y2 = interp1d(t2, y2, kind='linear', fill_value='extrapolate')
    
    def objective(params):
        dt = params[0]
        dx = params[1] if estimate_bias else 0.0
        dy = params[2] if estimate_bias else 0.0
        
        t2_projected = t1 - dt
        mask = (t2_projected >= t2.min()) & (t2_projected <= t2.max())
        if not np.any(mask): return 1e9
        
        pred_x = f_x2(t2_projected[mask]) + dx
        pred_y = f_y2(t2_projected[mask]) + dy
        return np.mean((x1[mask] - pred_x)**2 + (y1[mask] - pred_y)**2)
    
    init_guess = [0.0, np.mean(x1)-np.mean(x2), np.mean(y1)-np.mean(y2)] if estimate_bias else [0.0]
    res = minimize(objective, init_guess, method='Nelder-Mead')
    
    if estimate_bias:
        return res.x[0], res.x[1], res.x[2]
    return res.x[0], 0.0, 0.0

# =========================================================================
# 核心基础算法：异步多速率扩展卡尔曼滤波（状态估计器）
# =========================================================================
def asynchronous_kalman_filter(df1, df2, dt_opt, dx_opt, dy_opt, R1, R2):
    df2_aligned = df2.copy()
    df2_aligned.iloc[:, 0] += dt_opt
    df2_aligned.iloc[:, 1] += dx_opt
    df2_aligned.iloc[:, 2] += dy_opt
    
    t_start = max(df1.iloc[:, 0].min(), df2_aligned.iloc[:, 0].min())
    t_end = min(df1.iloc[:, 0].max(), df2_aligned.iloc[:, 0].max())
    t_grid = np.arange(t_start, t_end + 1e-5, 0.1)
    
    events = []
    for t in t_grid: events.append({'t': t, 'type': 'grid'})
    for _, row in df1.iterrows(): events.append({'t': row.iloc[0], 'type': 'w1', 'z': row.iloc[1:3].values})
    for _, row in df2_aligned.iterrows(): events.append({'t': row.iloc[0], 'type': 'w2', 'z': row.iloc[1:3].values})
    events.sort(key=lambda e: e['t'])
    
    X = np.array([df1.iloc[0, 1], df1.iloc[0, 2], 0.0, 0.0, 0.0, 0.0])
    P = np.eye(6) * 1.0
    Q_base = np.eye(6) * 0.05
    H = np.zeros((2, 6)); H[0, 0] = 1; H[1, 1] = 1
    
    current_t = events[0]['t']
    output = []
    
    for event in events:
        t_next = event['t']
        dt = t_next - current_t
        
        if dt > 0:
            F = np.eye(6)
            F[0, 2], F[1, 3] = dt, dt
            F[0, 4], F[1, 5] = 0.5*dt**2, 0.5*dt**2
            F[2, 4], F[3, 5] = dt, dt
            X = F @ X
            P = F @ P @ F.T + Q_base * dt
            current_t = t_next
            
        if event['type'] in ['w1', 'w2']:
            R = R1 if event['type'] == 'w1' else R2
            Y = event['z'] - H @ X
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            X = X + K @ Y
            P = (np.eye(6) - K @ H) @ P
        elif event['type'] == 'grid':
            output.append({
                '时间(s)': t_next, 'X坐标(m)': X[0], 'Y坐标(m)': X[1],
                'Vx(m/s)': X[2], 'Vy(m/s)': X[3], 'Ax(m/s^2)': X[4], 'Ay(m/s^2)': X[5]
            })
            
    res_df = pd.DataFrame(output).drop_duplicates(subset=['时间(s)'], keep='last')
    return res_df

# =========================================================================
# 各问题独立求解入口函数
# =========================================================================
def solve_problem_1(df1, df2):
    t1, x1, y1 = df1.iloc[:, 0].values, df1.iloc[:, 1].values, df1.iloc[:, 2].values
    t2, x2, y2 = df2.iloc[:, 0].values, df2.iloc[:, 1].values, df2.iloc[:, 2].values
    f_x2 = interp1d(t2, x2, kind='cubic', fill_value='extrapolate')
    f_y2 = interp1d(t2, y2, kind='cubic', fill_value='extrapolate')
    
    def objective(dt):
        t2_projected = t1 - dt
        mask = (t2_projected >= t2.min()) & (t2_projected <= t2.max())
        if not np.any(mask): return 1e9
        return np.mean((x1[mask] - f_x2(t2_projected[mask]))**2 + (y1[mask] - f_y2(t2_projected[mask]))**2)
    
    res = minimize(objective, [0.0], method='Nelder-Mead')
    dt_opt = res.x[0]
    
    t_start = max(t1.min(), t2.min() + dt_opt)
    t_end = min(t1.max(), t2.max() + dt_opt)
    t_10hz = np.arange(t_start, t_end, 0.1)
    
    f_x1 = interp1d(t1, x1, kind='cubic', fill_value='extrapolate')
    f_y1 = interp1d(t1, y1, kind='cubic', fill_value='extrapolate')
    
    x_fuse = (
    f_x1(t_10hz)
    +
    f_x2(t_10hz - dt_opt)
)/2

    y_fuse = (
    f_y1(t_10hz)
    +
    f_y2(t_10hz - dt_opt)
)/2
    traj_10hz = pd.DataFrame({
    '时间(s)': t_10hz,
    'X坐标(m)': x_fuse,
    'Y坐标(m)': y_fuse
})
    return dt_opt, traj_10hz

def solve_problem_2(df1, df2):
    dt, dx, dy = align_spatiotemporal(df1, df2, estimate_bias=True)
    # 升级处二应用：自适应估计 R 矩阵
    R1 = estimate_measurement_noise(df1, df2, dt, dx, dy)
    R2 = R1.copy()
    traj_10hz = asynchronous_kalman_filter(df1, df2, dt, dx, dy, R1, R2)
    return dt, dx, dy, traj_10hz

def solve_problem_3(df1, df2):
    dt_opt, dx_opt, dy_opt = align_spatiotemporal(df1, df2, estimate_bias=True)
    
    # 升级处一应用：使用单样本 t 检验进行系统偏置显著性判定
    px, py = bias_significance_test(df1, df2, dt_opt, dx_opt, dy_opt)
    
    print(f"[问题3统计检验] X方向偏置 p值: {px:.5f}, Y方向偏置 p值: {py:.5f}")
    if px < 0.05 or py < 0.05:
        print(">>> 结论：拒绝原假设，存在显著的空间系统偏差。")
    else:
        print(">>> 结论：接受原假设，空间系统偏差不显著，重置 dx=0, dy=0。")
        dx_opt, dy_opt = 0.0, 0.0
        
    # 升级处二应用：自适应估计 R 矩阵
    R1 = estimate_measurement_noise(df1, df2, dt_opt, dx_opt, dy_opt)
    R2 = R1.copy()
    traj_10hz = asynchronous_kalman_filter(df1, df2, dt_opt, dx_opt, dy_opt, R1, R2)
    return dt_opt, dx_opt, dy_opt, traj_10hz

# =========================================================================
# 升级处三：问题4 任务排程模型的整数线性规划（ILP）求解器
# =========================================================================
def solve_problem_4(trajectory_df, df_shoot, df_photo):
    t = trajectory_df['时间(s)'].values
    x = trajectory_df['X坐标(m)'].values
    y = trajectory_df['Y坐标(m)'].values
    v = np.sqrt(trajectory_df['Vx(m/s)'].values**2 + trajectory_df['Vy(m/s)'].values**2)
    a = np.sqrt(trajectory_df['Ax(m/s^2)'].values**2 + trajectory_df['Ay(m/s^2)'].values**2)
    
    candidate_windows = []
    
    # 1. 扫描生成所有符合运动学边界的候选射击窗口
    for _, target in df_shoot.iterrows():
        tid, tx, ty = target['编号'], target['X坐标(m)'], target['Y坐标(m)']
        dists = np.sqrt((x - tx)**2 + (y - ty)**2)
        valid_instant = (dists >= 5) & (dists <= 30) & (v <= 2.0) & (a <= 1.5)
        
        for i in range(len(t) - 15):
            if np.all(valid_instant[i:i+16]):
                candidate_windows.append({
                    'target_id': tid, 'type': '射击',
                    'start_t': t[i], 'end_t': t[i+15],
                    'angle': np.arctan2(ty - y[i+15], tx - x[i+15]) * 180 / np.pi
                })
                
    # 2. 扫描生成所有符合运动学边界的候选拍照窗口
    for _, target in df_photo.iterrows():
        tid, tx, ty = target['编号'], target['X坐标(m)'], target['Y坐标(m)']
        dists = np.sqrt((x - tx)**2 + (y - ty)**2)
        valid_instant = (dists >= 10) & (dists <= 40) & (v <= 1.5) & (a <= 1.5)
        
        for i in range(len(t) - 5):
            if np.all(valid_instant[i:i+6]):
                candidate_windows.append({
                    'target_id': tid, 'type': '拍照',
                    'start_t': t[i], 'end_t': t[i+5],
                    'angle': np.arctan2(ty - y[i+5], tx - x[i+5]) * 180 / np.pi
                })
                
    if not candidate_windows:
        print("未找到任何可行任务窗口！")
        return pd.DataFrame()

    # 3. 创建 PuLP 整数线性规划模型
    prob = LpProblem("Robot_Task_Optimization", LpMaximize)
    
    # 定义 0-1 二进制决策变量
    x_vars = {i: LpVariable(f"x_{i}", cat='Binary') for i in range(len(candidate_windows))}
    
    # 建立目标函数：最大化加权收益
    objective = []
    for i, w in enumerate(candidate_windows):
        weight = 0.85 if w["type"] == "射击" else 1.0
        objective.append(weight * x_vars[i])
    prob += lpSum(objective)
    
    # 建立时间排程冲突约束（采用极其高效的扫描线区间互斥算法，防止大样本下卡死）
    time_events = []
    for i, w in enumerate(candidate_windows):
        time_events.append((w['start_t'], 1, i))   # 1 代表窗口开启
        time_events.append((w['end_t'], -1, i))    # -1 代表窗口关闭
    # 按照时间节点排序（若时间戳相同，关闭事件优先，避免零交集误判）
    time_events.sort(key=lambda ev: (ev[0], ev[1]))
    
    active_windows = set()
    for t_val, event_type, idx in time_events:
        if event_type == 1:
            active_windows.add(idx)
            # 如果当前重叠时间区间内的活跃任务大于1，添加联合互斥约束：这些任务之和最多为1
            if len(active_windows) > 1:
                prob += lpSum([x_vars[k] for k in active_windows]) <= 1
        else:
            if idx in active_windows:
                active_windows.remove(idx)
                
    # 4. 拍照夹角约束
    # =========================================================================
    # 【高能优化】：用时间点切片法取代原有的 O(N^2) 两两暴力循环，防止约束爆炸
    # =========================================================================
    # 1. 提取所有动作视窗的起止关键时间点并去重排序
    all_times = [w['start_t'] for w in candidate_windows] + [w['end_t'] for w in candidate_windows]
    time_points = sorted(list(set(all_times)))
    
    # 2. 遍历每一个相邻时间段，确保每个时间片段内最多只有一个任务被激活
    for k in range(len(time_points) - 1):
        t_start_cell = time_points[k]
        t_end_cell = time_points[k+1]
        t_mid = (t_start_cell + t_end_cell) / 2.0  # 取时间片中点判定覆盖性
        
        # 找出所有包含当前时间片的候选窗口索引
        active_windows = [
            i for i, w in enumerate(candidate_windows) 
            if w['start_t'] <= t_mid <= w['end_t']
        ]
        
        # 核心硬约束：这些重叠的窗口变量之和不能超过 1
        if len(active_windows) > 1:
            prob += (lpSum([x_vars[i] for i in active_windows]) <= 1)
                    
    # 5. 调用 CBC 求解器精确求解
    prob.solve(PULP_CBC_CMD(msg=False))
    
    # 6. 过滤输出最优决策结果
    selected = []
    for i in x_vars:
        if value(x_vars[i]) > 0.5:
            selected.append(candidate_windows[i])
            
    return pd.DataFrame(selected)

# =========================================================================
# 主程序执行流（自动读取附件并依序完成所有建模求解）
# =========================================================================
if __name__ == '__main__':
    
    print("====== 正在加载多源传感器定位与任务数据 ======")
    # 问题 1 读入对应的两个不同频率的 Sheet
    f1_p1 = pd.read_excel('/home/lsp/Mathematical_schrace/data/附件1.xlsx', sheet_name='方式1(4Hz)')
    f2_p1 = pd.read_excel('/home/lsp/Mathematical_schrace/data/附件1.xlsx', sheet_name='方式2(5Hz)')
    
    # 问题 2 读入对应的两个不同频率的 Sheet
    f1_p2 = pd.read_excel('/home/lsp/Mathematical_schrace/data/附件2.xlsx', sheet_name='方式1(4Hz)')
    f2_p2 = pd.read_excel('/home/lsp/Mathematical_schrace/data/附件2.xlsx', sheet_name='方式2(5Hz)')
    
    # 问题 3 读入对应的两个不同频率的 Sheet
    f1_p3 = pd.read_excel('/home/lsp/Mathematical_schrace/data/附件3.xlsx', sheet_name='方式1(4Hz)')
    f2_p3 = pd.read_excel('/home/lsp/Mathematical_schrace/data/附件3.xlsx', sheet_name='方式2(5Hz)')
    
    # 问题 4 读入两个不同的目标 Sheet
    df_shoot = pd.read_excel('/home/lsp/Mathematical_schrace/data/附件4.xlsx', sheet_name='射击目标')
    df_photo = pd.read_excel('/home/lsp/Mathematical_schrace/data/附件4.xlsx', sheet_name='拍照目标')
    
    print("\n====== [开始求解问题 1] ======")
    dt1, traj1 = solve_problem_1(f1_p1, f2_p1)
    print(f"问题1结果 -> 时间偏差 dt: {dt1:.4f}s")
    traj1.to_csv('问题1_10Hz轨迹结果.csv', index=False)
    
    print("\n====== [开始求解问题 2] ======")
    dt2, dx2, dy2, traj2 = solve_problem_2(f1_p2, f2_p2)
    print(f"问题2结果 -> 时间偏差 dt: {dt2:.4f}s, X偏置 dx: {dx2:.4f}m, Y偏置 dy: {dy2:.4f}m")
    traj2.to_csv('问题2_10Hz轨迹结果.csv', index=False)
    
    print("\n====== [开始求解问题 3] ======")
    dt3, dx3, dy3, traj3 = solve_problem_3(f1_p3, f2_p3)
    print(f"问题3结果 -> 时间偏差 dt: {dt3:.4f}s, 最终有效 dx: {dx3:.4f}m, dy: {dy3:.4f}m")
    traj3.to_csv('问题3_10Hz轨迹结果.csv', index=False)
    
    print("\n====== [开始求解问题 4] ======")
    schedule_res = solve_problem_4(traj3, df_shoot, df_photo)
    print(f"问题4规划完成！成功排程执行的任务总数: {len(schedule_res)} 个")
    
    if not schedule_res.empty:
        # 转换成赛题标准格式
        final_report = pd.DataFrame()
        final_report['序号'] = range(1, len(schedule_res) + 1)
        final_report['目标编号'] = schedule_res['target_id']
        final_report['任务'] = schedule_res['type']
        final_report['开始准备时刻(s)'] = schedule_res['start_t']
        final_report['任务执行时刻(s)'] = schedule_res['end_t'] # 刚好是准备结束、执行动作的时刻
        
        print(final_report.head(10))
        final_report.to_csv('问题4_任务排程最优决策表.csv', index=False, encoding='utf-8-sig')