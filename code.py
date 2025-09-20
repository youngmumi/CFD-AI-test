# ==============================
# Enhanced GPR Airfoil Analysis with Improved Visualizations
# ==============================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# -----------------------------
# 1. Enhanced Surrogate Data Generation
# -----------------------------
def generate_airfoil_data():
    """Generate realistic airfoil performance data"""
    AoA_deg = np.linspace(0, 15, 50)      # 받음각 (Angle of Attack)
    Velocity = np.linspace(20, 60, 10)    # 속도 m/s
    rho = 1.225  # 공기 밀도 [kg/m3]
    S = 1.0      # 기준 면적 [m2]
    
    # 개선된 공기역학 모델 (비선형성 추가)
    C_D0 = 0.012
    k = 0.045
    
    data = []
    for V in Velocity:
        for AoA in AoA_deg:
            alpha = np.radians(AoA)
            
            # 실제 airfoil 특성에 더 가까운 모델링
            C_L = 2 * np.pi * alpha * (1 - 0.1 * (alpha/np.pi)**2)  # 비선형성 추가
            C_D = C_D0 + k * C_L**2 + 0.001 * alpha**3  # 고차항 추가
            
            # Reynolds 수 효과 추가 (간단한 모델)
            Re = rho * V * 1.0 / 1.81e-5  # 동적 점성도
            Re_factor = 1 + 0.05 * np.log10(Re/1e6)
            C_L *= Re_factor
            
            L_to_D = C_L / C_D if C_D > 0 else 0
            q = 0.5 * rho * V**2
            Lift = C_L * q * S
            Drag = C_D * q * S
            
            # 노이즈 추가 (실험 데이터의 불확실성 모사)
            noise_factor = 0.02
            Lift += np.random.normal(0, noise_factor * abs(Lift))
            Drag += np.random.normal(0, noise_factor * abs(Drag))
            L_to_D = Lift / Drag if Drag > 0 else 0
            
            data.append([AoA, V, Lift, Drag, L_to_D, C_L, C_D])
    
    df = pd.DataFrame(data, columns=["AoA_deg","Velocity_mps","Lift_N","Drag_N","L_to_D","C_L","C_D"])
    return df

# 데이터 생성
print("🔄 Surrogate 데이터 생성 중...")
df = generate_airfoil_data()
df.to_csv("enhanced_surrogate_airfoil.csv", index=False)
print(f"✅ Surrogate 데이터 생성 완료, 총 샘플 수: {len(df)}")

# 기본 통계 출력
print("\n📊 데이터 기본 통계:")
print(df.describe().round(4))

# -----------------------------
# 2️. 개선된 데이터 전처리
# -----------------------------
def prepare_data(df, target_columns=["Lift_N","Drag_N","L_to_D"]):
    """데이터 전처리 및 분리"""
    X = df[["AoA_deg","Velocity_mps"]].values
    y = df[target_columns].values
    
    # 스케일링
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y)
    
    # Train/Test 분리
    X_train, X_test, y_train, y_test, y_train_raw, y_test_raw = train_test_split(
        X_scaled, y_scaled, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test, y_train_raw, y_test_raw, scaler_X, scaler_y

X_train, X_test, y_train, y_test, y_train_raw, y_test_raw, scaler_X, scaler_y = prepare_data(df)
print("✅ 데이터 전처리 및 Train/Test 분리 완료")

# -----------------------------
# 3. 개선된 GPR 모델 학습
# -----------------------------
def create_optimized_kernel():
    """최적화된 GPR 커널 생성"""
    # 더 안정적인 커널 설정
    kernel = (C(1.0, (1e-3, 1e3)) * 
              RBF(length_scale=[1.0, 1.0], length_scale_bounds=(1e-2, 1e2)) + 
              WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-6, 1e-1)))
    return kernel

def train_gpr_models(X_train, y_train):
    """GPR 모델들 학습"""
    kernel = create_optimized_kernel()
    models = []
    target_names = ["Lift_N", "Drag_N", "L_to_D"]
    
    print("🔄 GPR 모델 학습 중...")
    for i in range(y_train.shape[1]):
        gpr = GaussianProcessRegressor(
            kernel=kernel, 
            n_restarts_optimizer=10,
            random_state=42,
            alpha=1e-6  # 수치 안정성 향상
        )
        gpr.fit(X_train, y_train[:,i])
        models.append(gpr)
        print(f"✅ {target_names[i]} GPR 모델 학습 완료")
    
    return models

models = train_gpr_models(X_train, y_train)

# -----------------------------
# 4 모델 성능 평가
# -----------------------------
def evaluate_models(models, X_test, y_test, y_test_raw, scaler_y):
    """모델 성능 평가"""
    target_names = ["Lift_N", "Drag_N", "L_to_D"]
    results = {}
    
    print("\n📈 모델 성능 평가:")
    print("-" * 50)
    
    for i, (model, name) in enumerate(zip(models, target_names)):
        # 예측
        y_pred_scaled, y_std_scaled = model.predict(X_test, return_std=True)
        
        # 원래 스케일로 복원
        dummy = np.zeros((len(y_pred_scaled), len(target_names)))
        dummy[:, i] = y_pred_scaled
        y_pred_full = scaler_y.inverse_transform(dummy)
        y_pred = y_pred_full[:, i]
        y_std = y_std_scaled * scaler_y.scale_[i]
        
        # 성능 지표 계산
        mse = mean_squared_error(y_test_raw[:, i], y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test_raw[:, i], y_pred)
        
        results[name] = {
            'y_pred': y_pred,
            'y_std': y_std,
            'y_true': y_test_raw[:, i],
            'mse': mse,
            'rmse': rmse,
            'r2': r2
        }
        
        print(f"{name:>8}: RMSE={rmse:.4f}, R²={r2:.4f}")
    
    return results

results = evaluate_models(models, X_test, y_test, y_test_raw, scaler_y)

# -----------------------------
# 5️ 향상된 시각화
# -----------------------------
def create_comprehensive_plots(results, X_test, scaler_X):
    """종합적인 시각화"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('GPR Airfoil Performance Prediction Analysis', fontsize=16, fontweight='bold')
    
    target_names = ["Lift_N", "Drag_N", "L_to_D"]
    colors = ['blue', 'red', 'green']
    
    # 첫 번째 행: 예측 vs 실제값 비교
    for i, (name, color) in enumerate(zip(target_names, colors)):
        ax = axes[0, i]
        result = results[name]
        
        # 불확실성과 함께 예측값 시각화
        indices = range(len(result['y_pred']))
        ax.errorbar(indices, result['y_pred'], yerr=result['y_std'], 
                   fmt='o', alpha=0.6, color=color, label=f'Predicted ±σ')
        ax.scatter(indices, result['y_true'], color='black', s=30, 
                  alpha=0.7, label='True Values')
        
        ax.set_xlabel('Test Sample Index')
        ax.set_ylabel(name.replace('_', ' '))
        ax.set_title(f'{name} Prediction vs True\nR²={result["r2"]:.4f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 두 번째 행: 상관관계 및 잔차 분석
    for i, (name, color) in enumerate(zip(target_names, colors)):
        ax = axes[1, i]
        result = results[name]
        
        if i < 2:  # Lift와 Drag는 상관관계 플롯
            ax.scatter(result['y_true'], result['y_pred'], alpha=0.6, color=color)
            
            # 완벽한 예측 라인
            min_val = min(result['y_true'].min(), result['y_pred'].min())
            max_val = max(result['y_true'].max(), result['y_pred'].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, label='Perfect Prediction')
            
            ax.set_xlabel(f'True {name}')
            ax.set_ylabel(f'Predicted {name}')
            ax.set_title(f'{name} Correlation\nRMSE={result["rmse"]:.4f}')
            ax.legend()
        else:  # L/D는 입력 공간에서의 분포
            X_test_orig = scaler_X.inverse_transform(X_test)
            scatter = ax.scatter(X_test_orig[:, 0], X_test_orig[:, 1], 
                               c=result['y_pred'], cmap='viridis', alpha=0.7)
            ax.set_xlabel('Angle of Attack (deg)')
            ax.set_ylabel('Velocity (m/s)')
            ax.set_title('L/D Prediction Distribution')
            plt.colorbar(scatter, ax=ax, label='Predicted L/D')
        
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

create_comprehensive_plots(results, X_test, scaler_X)

# -----------------------------
# 6️ 예측 표면 시각화
# -----------------------------
def plot_prediction_surface():
    """L/D 예측 표면 시각화"""
    # 그리드 생성
    aoa_range = np.linspace(0, 15, 30)
    vel_range = np.linspace(20, 60, 25)
    AoA_grid, Vel_grid = np.meshgrid(aoa_range, vel_range)
    
    # 예측을 위한 입력 준비
    X_grid = np.column_stack([AoA_grid.ravel(), Vel_grid.ravel()])
    X_grid_scaled = scaler_X.transform(X_grid)
    
    # L/D 예측 (index 2)
    y_pred_scaled, y_std_scaled = models[2].predict(X_grid_scaled, return_std=True)
    
    # 스케일 복원
    dummy = np.zeros((len(y_pred_scaled), 3))
    dummy[:, 2] = y_pred_scaled
    y_pred_full = scaler_y.inverse_transform(dummy)
    L_D_pred = y_pred_full[:, 2].reshape(AoA_grid.shape)
    L_D_std = (y_std_scaled * scaler_y.scale_[2]).reshape(AoA_grid.shape)
    
    # 시각화
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # L/D 예측 표면
    contour1 = ax1.contourf(AoA_grid, Vel_grid, L_D_pred, levels=20, cmap='viridis')
    ax1.contour(AoA_grid, Vel_grid, L_D_pred, levels=10, colors='black', alpha=0.3, linewidths=0.5)
    plt.colorbar(contour1, ax=ax1, label='L/D')
    ax1.set_xlabel('Angle of Attack (deg)')
    ax1.set_ylabel('Velocity (m/s)')
    ax1.set_title('L/D Prediction Surface')
    
    # 불확실성 표면
    contour2 = ax2.contourf(AoA_grid, Vel_grid, L_D_std, levels=20, cmap='Reds')
    plt.colorbar(contour2, ax=ax2, label='Prediction Uncertainty (σ)')
    ax2.set_xlabel('Angle of Attack (deg)')
    ax2.set_ylabel('Velocity (m/s)')
    ax2.set_title('Prediction Uncertainty Surface')
    
    # 테스트 포인트 오버레이
    X_test_orig = scaler_X.inverse_transform(X_test)
    ax1.scatter(X_test_orig[:, 0], X_test_orig[:, 1], c='red', s=20, alpha=0.7, label='Test Points')
    ax2.scatter(X_test_orig[:, 0], X_test_orig[:, 1], c='white', s=20, alpha=0.7, label='Test Points')
    ax1.legend()
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

plot_prediction_surface()

# -----------------------------
# 7️ 결과 요약 및 저장
# -----------------------------
def save_results(results, df):
    """결과 저장"""
    # 성능 요약
    summary = pd.DataFrame({
        'Target': list(results.keys()),
        'RMSE': [results[k]['rmse'] for k in results.keys()],
        'R²': [results[k]['r2'] for k in results.keys()],
        'MSE': [results[k]['mse'] for k in results.keys()]
    })
    
    summary.to_csv('gpr_performance_summary.csv', index=False)
    print("\n💾 결과 요약:")
    print(summary.to_string(index=False, float_format='%.6f'))
    
    print(f"\n📁 파일 저장:")
    print("- enhanced_surrogate_airfoil.csv: 원본 데이터")
    print("- gpr_performance_summary.csv: 성능 요약")

save_results(results, df)

print("\n분석 완료! GPR을 이용한 airfoil 성능 예측 모델이 성공적으로 구축되었습니다.")
