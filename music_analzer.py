# music_analyzer.py (最终修复版)

import sys
import os
import warnings
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf
from pydub import AudioSegment
from typing import Dict, Any, Optional, List
import colorsys
import random
from matplotlib.patches import Circle, Ellipse, Rectangle, Polygon, RegularPolygon, PathPatch
from matplotlib.path import Path
from matplotlib.collections import PatchCollection
from matplotlib.colors import LinearSegmentedColormap

# 配置警告忽略
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# ================= 辅助函数 =================
def convert_to_wav_if_needed(file_path: str) -> str:
    """如果文件不是WAV格式，则转换为WAV格式"""
    if not file_path.lower().endswith('.wav'):
        print(f"转换文件为WAV格式: {file_path}")
        output_path = file_path.rsplit('.', 1)[0] + '.wav'

        try:
            # 尝试使用soundfile读取并写入WAV
            data, samplerate = sf.read(file_path)
            sf.write(output_path, data, samplerate)
            print(f"转换完成: {output_path}")
            return output_path
        except:
            # 如果soundfile失败，使用pydub
            try:
                audio = AudioSegment.from_file(file_path)
                audio.export(output_path, format="wav")
                print(f"使用pydub转换完成: {output_path}")
                return output_path
            except Exception as e:
                print(f"文件转换失败: {e}. 尝试直接使用原文件.")
                return file_path
    return file_path


# ================= 音乐分析函数 =================
def analyze_music(file_path: str) -> Optional[Dict[str, Any]]:
    """分析音乐文件的特征"""
    print(f"开始分析: {file_path}")
    start_time = time.time()

    try:
        # 1. 检查文件是否存在
        if not os.path.exists(file_path):
            print(f"错误: 文件不存在 - {file_path}")
            return None

        print("加载音频文件...")
        # 2. 尝试加载音频文件
        try:
            y, sr = librosa.load(file_path, sr=None, mono=True, duration=60)  # 只加载前1分钟
        except Exception as load_error:
            print(f"音频加载失败: {load_error}")
            # 尝试使用soundfile作为备用方案
            try:
                data, sr = sf.read(file_path)
                if len(data.shape) > 1:  # 如果是立体声，转换为单声道
                    y = librosa.to_mono(data.T)
                else:
                    y = data
                print(f"使用soundfile加载成功")
            except Exception as sf_error:
                print(f"备选加载方案也失败: {sf_error}")
                return None

        duration = len(y) / sr
        print(f"音频时长: {duration:.1f}秒")

        print("分析节奏...")
        # 3. 分析节奏 - 添加额外错误处理
        try:
            tempo_result = librosa.beat.beat_track(y=y, sr=sr)

            # 处理不同版本的返回类型
            if isinstance(tempo_result, tuple):
                if len(tempo_result) == 2:
                    # 修复NumPy兼容性问题：提取标量值
                    if isinstance(tempo_result[0], np.ndarray):
                        tempo = float(tempo_result[0].item())
                    else:
                        tempo = float(tempo_result[0])
                    beat_frames = tempo_result[1]
                else:
                    raise ValueError("意外的节拍分析结果格式")
            else:
                if isinstance(tempo_result, np.ndarray):
                    tempo = float(tempo_result.item())
                else:
                    tempo = float(tempo_result)
                beat_frames = np.array([])

            # 确保tempo是有效值
            if tempo <= 0 or np.isnan(tempo):
                tempo = 120.0
                print("节奏分析失败，使用默认BPM: 120")

            beat_times = librosa.frames_to_time(beat_frames, sr=sr)
            if len(beat_times) == 0:
                # 生成模拟节拍
                beat_times = np.arange(0, duration, 60.0 / tempo)
        except Exception as beat_error:
            print(f"节拍分析错误: {beat_error}")
            # 使用安全的默认值
            tempo = 120.0
            beat_times = np.arange(0, min(30, duration), 0.5)  # 每0.5秒一个节拍

        print(f"检测到的BPM: {tempo:.1f}")

        print("分析音量...")
        # 4. 分析音量
        try:
            rms = librosa.feature.rms(y=y)
            rms_db = librosa.amplitude_to_db(rms, ref=0)
        except Exception as vol_error:
            print(f"音量分析错误: {vol_error}")
            # 创建模拟音量数据
            rms_db = np.array([[-30, -20, -15, -10, -5]] * 10).T

        # 音量特征计算
        if rms_db.size > 0:
            peak_vol = np.max(rms_db)
            avg_vol = np.mean(rms_db)
            var_vol = np.var(rms_db)
        else:
            peak_vol = 80
            avg_vol = 70
            var_vol = 10

        # 修复这一行语法错误 - 使用英文字符
        print(f"平均音量: {avg_vol:.1f} dB, 峰值音量: {peak_vol:.1f} dB")

        print("分析调性...")
        # 5. 分析调性
        try:
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
            chroma_avg = np.mean(chroma, axis=1)
            key_index = np.argmax(chroma_avg)
            notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            key = notes[key_index]

            # 判断大调小调
            major_minor_score = np.mean(chroma[key_index]) - np.mean(chroma[(key_index + 3) % 12])
            key_mode = "大调" if major_minor_score > 0 else "小调"
            key_info = f"{key} {key_mode}"
        except Exception as key_error:
            print(f"调性分析错误: {key_error}")
            key_info = "C 大调"  # 默认调性

        print(f"检测到的调性: {key_info}")

        # 6. 分析结果
        analysis_time = time.time() - start_time
        result = {
            "tempo": tempo,
            "key": key_info,
            "volume_profile": {
                "peak": peak_vol,
                "avg": avg_vol,
                "variance": var_vol
            },
            "beat_times": beat_times.tolist(),
            "analysis_time": analysis_time,
            "duration": duration
        }

        return result

    except Exception as e:
        print(f"分析过程发生严重错误: {e}")
        import traceback
        traceback.print_exc()
        return None


# ================= 可视化函数 =================
def display_results(results: Dict[str, Any]):
    """以文本形式显示分析结果"""
    print("\n" + "=" * 50)
    print("音乐分析结果:")
    print("=" * 50)
    print(f"节奏(BPM): {results['tempo']:.1f}")
    print(f"调性: {results['key']}")
    print(f"音量特征:")
    print(f"  峰值音量: {results['volume_profile']['peak']:.1f} dB")
    print(f"  平均音量: {results['volume_profile']['avg']:.1f} dB")
    print(f"  音量波动: {results['volume_profile']['variance']:.1f}")
    print(f"检测到节拍数: {len(results['beat_times'])}")
    print(f"分析用时: {results['analysis_time']:.2f}秒")
    print("=" * 50)


def visualize_analysis(file_path: str, results: Dict[str, Any]):
    """创建音乐分析的可视化图表"""
    # 确保输出目录存在
    output_dir = "analysis_results"
    os.makedirs(output_dir, exist_ok=True)

    # 加载音频文件
    try:
        y, sr = librosa.load(file_path)
    except Exception as e:
        print(f"无法加载文件用于可视化: {e}")
        return

    # 创建波形图
    plt.figure(figsize=(14, 8))

    # 1. 波形图
    plt.subplot(3, 1, 1)
    librosa.display.waveshow(y, sr=sr, alpha=0.6)
    plt.title('音频波形', fontsize=14)
    plt.xlabel('')

    # 2. 频谱图
    plt.subplot(3, 1, 2)
    try:
        S = librosa.feature.melspectrogram(y=y, sr=sr)
        S_db = librosa.power_to_db(S, ref=np.max)
        librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title('梅尔频谱图', fontsize=14)
    except Exception as e:
        print(f"频谱图创建失败: {e}")

    # 3. 节拍图
    plt.subplot(3, 1, 3)
    try:
        beat_times = np.array(results['beat_times'])
        plt.vlines(beat_times, 0, 1, color='r', linestyle='--', alpha=0.8)
        plt.title(f'节拍点检测: {len(beat_times)} 拍 (BPM: {results["tempo"]:.1f})', fontsize=14)
        plt.xlabel('时间 (秒)')
        plt.yticks([])
        plt.ylim(0, 1)
    except Exception as e:
        print(f"节拍图创建失败: {e}")

    # 保存图表
    plt.tight_layout()
    viz_path = os.path.join(output_dir, "audio_analysis.png")
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"音频分析图已保存到: {viz_path}")


# ================= 装饰画生成函数 =================
def generate_music_art(data: Dict[str, Any], save_path: str = "music_art.png",
                       width: int = 1200, height: int = 1200, dpi: int = 150) -> str:
    """
    根据音乐分析数据生成彩色装饰画

    参数:
        data: 音乐分析结果字典
        save_path: 保存图片的路径
        width: 图像宽度（像素）
        height: 图像高度（像素）
        dpi: 图像分辨率

    返回:
        保存的文件路径
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 创建图像和画布
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi, facecolor='#111111')
    ax = fig.add_subplot(111)

    # 设置画布范围和外观
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')  # 隐藏坐标轴

    # 1. 根据调性确定主色调
    key = data['key'].split()[0]  # 提取音符部分（如"C"）
    key_to_hue = {
        'C': 0 / 12, 'C#': 1 / 12, 'Db': 1 / 12,
        'D': 2 / 12, 'D#': 3 / 12, 'Eb': 3 / 12,
        'E': 4 / 12, 'F': 5 / 12, 'F#': 6 / 12, 'Gb': 6 / 12,
        'G': 7 / 12, 'G#': 8 / 12, 'Ab': 8 / 12,
        'A': 9 / 12, 'A#': 10 / 12, 'Bb': 10 / 12,
        'B': 11 / 12
    }

    # 如果没有识别到调性，使用随机色调
    base_hue = key_to_hue.get(key, random.random())

    # 2. 音量映射
    vol_avg = data['volume_profile']['avg']
    vol_peak = data['volume_profile']['peak']
    vol_var = data['volume_profile']['variance']

    # 饱和度：平均音量映射 (30% 到 100%)
    saturation = np.clip((vol_avg + 30) / 20, 0.3, 1.0)

    # 明度：峰值音量映射 (40% 到 90%)
    brightness = np.clip((vol_peak + 10) / 40 + 0.4, 0.4, 0.9)

    # 3. 根据BPM确定元素密度
    bpm = data['tempo']
    base_element_count = 50  # 基础元素数量
    element_count = base_element_count + int(bpm / 2)  # 每2BPM增加1个元素

    # 4. 根据音量波动确定元素大小变化
    size_variation = np.clip(vol_var / 50, 0.1, 0.8)

    # 创建图形元素集合
    all_patches = []
    all_colors = []

    # 5. 创建背景渐变层
    bg_start_color = colorsys.hsv_to_rgb(
        (base_hue + 0.5) % 1.0,
        saturation * 0.3,
        brightness * 0.2
    )

    bg_end_color = colorsys.hsv_to_rgb(
        (base_hue + 0.7) % 1.0,
        saturation * 0.5,
        brightness * 0.3
    )

    # 创建渐变背景
    bg_cmap = LinearSegmentedColormap.from_list(
        'bg_gradient',
        [bg_start_color, bg_end_color]
    )
    bg_gradient = np.outer(
        np.linspace(0, 1, 100),
        np.linspace(0, 1, 100)
    )

    ax.imshow(
        bg_gradient,
        extent=[0, 1, 0, 1],
        cmap=bg_cmap,
        alpha=0.7,
        zorder=-100
    )

    # 6. 添加节拍点作为核心视觉元素
    beat_times = data.get('beat_times', [])
    duration = data.get('duration', len(beat_times) if beat_times else 60)

    max_time = max(beat_times) if beat_times else duration

    for i, beat_time in enumerate(beat_times):
        # 只处理前20个节拍点以避免过多
        if i >= 20:
            break

        # 计算归一化时间位置 (0-1)
        normalized_time = beat_time / max_time if max_time > 0 else i / 20

        # 基础元素位置 (在画布中部区域)
        center_x = 0.2 + normalized_time * 0.6
        center_y = 0.5 + 0.3 * np.sin(normalized_time * 8 * np.pi)  # 正弦波位置

        # 元素尺寸 - 基础大小加上随机和节拍位置影响
        base_size = 0.03 + 0.02 * (1 - normalized_time)  # 时间越晚尺寸越小
        size_factor = 1.0 + random.uniform(-size_variation, size_variation)
        size = base_size * size_factor

        # 创建形状元素 - 随机选择形状
        shape_type = random.choices(
            ['circle', 'polygon', 'ellipse', 'star', 'ring'],
            weights=[0.4, 0.2, 0.2, 0.1, 0.1],
            k=1
        )[0]

        if shape_type == 'circle':
            patch = Circle((center_x, center_y), size, edgecolor=None)
            patch.set_zorder(20 + i)
        elif shape_type == 'polygon':
            sides = random.randint(3, 8)
            patch = RegularPolygon(
                (center_x, center_y),
                sides,
                radius=size,
                orientation=random.uniform(0, 2 * np.pi)
            )
            patch.set_zorder(20 + i)
        elif shape_type == 'star':
            points = random.randint(5, 10)
            patch = RegularPolygon(
                (center_x, center_y),
                points,
                radius=size,
                orientation=random.uniform(0, 2 * np.pi),
                alpha=0.8
            )
            patch.set_zorder(20 + i)
        elif shape_type == 'ellipse':
            width_val = size * random.uniform(0.8, 1.2)
            height_val = size * random.uniform(0.5, 1.5)  # 修复这一行
            angle = random.uniform(0, 360)
            patch = Ellipse((center_x, center_y), width_val, height_val, angle=angle)
            patch.set_zorder(20 + i)
        else:  # ring
            # 外环
            outer_patch = Circle(
                (center_x, center_y),
                size,
                fill=False,
                linewidth=size * 200,  # 线宽基于元素大小
                alpha=0.7
            )
            outer_patch.set_zorder(29 + i)
            all_patches.append(outer_patch)

            # 内部小圆
            inner_size = size * random.uniform(0.3, 0.7)
            inner_patch = Circle(
            (center_x, center_y),
            inner_size,
            fill = True,
            alpha = 0.9
            )
            inner_patch.set_zorder(30 + i)
            all_patches.append(inner_patch)

            # 计算内部圆的颜色 (稍亮)
            inner_hue = (base_hue + 0.05) % 1.0
            inner_saturation = saturation * 1.1
            inner_brightness = brightness * 1.05
            inner_color = colorsys.hsv_to_rgb(
            inner_hue,
            np.clip(inner_saturation, 0, 1),
            np.clip(inner_brightness, 0, 1)  # 修复这一行
            )
            all_colors.append(inner_color)

            # 为外环添加颜色
            outer_hue = (base_hue + 0.1) % 1.0
            outer_color = colorsys.hsv_to_rgb(
            outer_hue,
            saturation * 0.9,
            brightness * 0.95
            )
            all_colors.append(outer_color)
            continue

        all_patches.append(patch)

        # 为节拍点元素生成颜色 - 偏移基础色调
        hue_offset = (i % 3) * 0.05  # 每三个元素循环一次色调
        element_hue = (base_hue + hue_offset + random.uniform(-0.03, 0.03)) % 1.0
        element_saturation = saturation * random.uniform(0.9, 1.1)
        element_brightness = brightness * random.uniform(0.9, 1.05)

        # 转换HSV到RGB
        color = colorsys.hsv_to_rgb(
            element_hue,
            np.clip(element_saturation, 0, 1),
            np.clip(element_brightness, 0, 1)  # 修复这一行
        )

        all_colors.append(color)

    # 7. 添加背景元素
    for i in range(element_count):
        # 随机位置
        x = random.random()
        y = random.random()

        # 元素形状 - 随机选择
        shape_type = random.choices(
            ['circle', 'ellipse', 'rectangle', 'triangle'],
            weights=[0.4, 0.3, 0.2, 0.1],
            k=1
        )[0]

        # 基础尺寸
        base_size = random.uniform(0.01, 0.05) * (1 - y)  # 上小下大

        # 尺寸变化 - 基于音量波动
        size_factor = 1.0 + random.uniform(-size_variation, size_variation)
        size = base_size * size_factor

        if shape_type == 'circle':
            patch = Circle((x, y), size)
            patch.set_zorder(i % 10)
        elif shape_type == 'ellipse':
            width_val = size * random.uniform(0.8, 1.2)
            height_val = size * random.uniform(0.5, 1.5)
            angle = random.uniform(0, 360)
            patch = Ellipse((x, y), width_val, height_val, angle=angle)
            patch.set_zorder(i % 10)
        elif shape_type == 'rectangle':
            width_val = size
            height_val = size * random.uniform(0.5, 2)
            angle = random.uniform(0, 90)
            patch = Rectangle(
                (x - width_val / 2, y - height_val / 2),
                width_val, height_val,
                angle=angle
            )
            patch.set_zorder(i % 10)
        else:  # triangle
            verts = [
                (x, y + size),
                (x - size, y - size),
                (x + size, y - size)
            ]
            patch = Polygon(verts)
            patch.set_zorder(i % 10)

        all_patches.append(patch)

        # 生成背景元素颜色 - 更透明，色调更分散
        hue_offset = (i % 12) * 0.08  # 创建色调变化
        element_hue = (base_hue + hue_offset + random.uniform(-0.1, 0.1)) % 1.0
        element_saturation = saturation * random.uniform(0.7, 0.9)
        element_brightness = brightness * random.uniform(0.8, 1.0)

        # 添加透明度 (20%-60%)
        alpha = random.uniform(0.2, 0.6)

        # 转换为RGB并添加透明度
        color = colorsys.hsv_to_rgb(
            element_hue,
            np.clip(element_saturation, 0, 1),
            np.clip(element_brightness, 0, 1)  # 修复这一行
        )
        color_with_alpha = color + (alpha,)

        all_colors.append(color_with_alpha)

    # 8. 添加装饰性线条（修复版本 - 使用PathPatch）
    for i in range(8):  # 添加8条装饰性线条
        # 基于节拍时间创建波浪线
        wave_points = []
        num_points = 50

        amplitude = 0.1 * (1 - i / 7)  # 逐渐减小的振幅
        for p in range(num_points):
            x = p / (num_points - 1)

            # 主要波形 (低频)
            y1 = 0.5 * amplitude * np.sin(2 * np.pi * (0.5 + i * 0.2) * x + i)

            # 次要波形 (高频)
            y2 = 0.3 * amplitude * np.sin(2 * np.pi * (5 + i * 0.5) * x + i * 0.7)

            wave_points.append((x, 0.5 + y1 + y2))

        # 创建路径对象
        path_verts = np.array(wave_points)
        path_codes = np.ones(len(path_verts), dtype=Path.code_type) * Path.LINETO
        path_codes[0] = Path.MOVETO

        # 创建Path对象
        path = Path(path_verts, path_codes)

        # 创建带线宽的路径Patch - 修复了线条兼容性问题
        linewidth = (i + 1) * 0.8
        line_alpha = 0.3

        patch = PathPatch(
            path,
            linewidth=linewidth,
            fill=False,  # 不填充内部
            alpha=line_alpha
        )
        patch.set_zorder(5 + i)

        all_patches.append(patch)

        # 线条颜色 - 基于基础色调
        line_hue = (base_hue + i * 0.05) % 1.0
        line_saturation = saturation * 0.7
        line_brightness = brightness * 0.9

        line_color = colorsys.hsv_to_rgb(
            line_hue,
            np.clip(line_saturation, 0, 1),
            np.clip(line_brightness, 0, 1)  # 修复这一行
        ) + (0.4,)  # 添加透明度

        all_colors.append(line_color)

    # 9. 创建图形集合并添加到画布
    if all_patches:  # 确保有元素可以显示
        # 关键修复：使用PatchCollection时不再传入zorder参数
        collection = PatchCollection(
            all_patches,
            match_original=True
        )

        # 设置颜色
        facecolors = []
        for color in all_colors:
            if len(color) == 3:  # RGB
                facecolors.append(color)
            else:  # RGBA
                facecolors.append(color)

        collection.set_facecolor(facecolors)
        collection.set_edgecolor(None)  # 无边框

        ax.add_collection(collection)

    # 10. 添加标题信息
    bpm_text = f"{bpm:.0f} BPM"
    key_text = f"{data['key']}"

    # 标题颜色 - 基于基础色调但更亮
    title_hue = (base_hue + 0.05) % 1.0
    title_color = colorsys.hsv_to_rgb(title_hue, 0.8, 0.95)

    # 在左上角添加标题
    ax.text(
        0.05, 0.95,
        key_text,
        color=title_color,
        fontsize=min(28, width / 30),
        ha='left', va='top',
        fontfamily='sans-serif',
        fontweight='bold',
        transform=ax.transAxes,
        zorder=1000  # 确保在最上方
    )

    # 在右上角添加BPM
    ax.text(
        0.95, 0.95,
        bpm_text,
        color=title_color,
        fontsize=min(26, width / 35),
        ha='right', va='top',
        fontfamily='monospace',
        fontweight='bold',
        transform=ax.transAxes,
        zorder=1000  # 确保在最上方
    )

    # 11. 添加小文本信息（右下角）
    meta_text = f"Generated from Audio Analysis"
    ax.text(
        0.98, 0.02,
        meta_text,
        color='#BBBBBB',
        fontsize=min(14, width / 80),
        ha='right', va='bottom',
        fontfamily='sans-serif',
        alpha=0.7,  # 修复这一行
        transform=ax.transAxes,
        zorder=1000  # 确保在最上方
    )

    # 12. 保存图像
    try:
        plt.savefig(
            save_path,
            dpi=dpi,
            bbox_inches='tight',
            pad_inches=0.1,
            facecolor='#111111'
        )
        print(f"装饰画成功保存到: {save_path}")
    except Exception as save_error:
        print(f"装饰画保存失败: {save_error}")
        # 尝试使用默认文件名
        alt_path = "music_art.png"
        plt.savefig(
            alt_path,
            dpi=dpi,
            bbox_inches='tight',
            pad_inches=0.1,
            facecolor='#111111'
        )
        print(f"装饰画已保存到备用路径: {alt_path}")
        save_path = alt_path

    plt.close(fig)

    return save_path

# ================= 主程序 =================
if __name__ == "__main__":
    print("=" * 60)
    print("音乐特征分析及可视化程序 - 最终修复版")
    print("=" * 60)

    try:
        # 定义音频文件路径
        music_file = r"C:\Users\HS\Desktop\测试\测试2.wav"

        print(f"分析文件: {music_file}")

        # 检查文件是否存在
        if not os.path.exists(music_file):
            print("错误: 文件不存在")
            alt_path = r"C:\Users\HS\Desktop\测试2.wav"
            if os.path.exists(alt_path):
                print(f"找到备用文件: {alt_path}")
                music_file = alt_path
            else:
                raise FileNotFoundError("未找到音频文件")

        # 转换文件格式（如果需要）
        converted_file = convert_to_wav_if_needed(music_file)
        print(f"使用文件: {converted_file}")

        # 执行分析
        print("\n开始音乐分析...")
        music_results = analyze_music(converted_file)

        # 处理分析结果
        if music_results is not None:
            # 1. 显示分析结果
            display_results(music_results)

            # 2. 创建传统可视化图表
            try:
                visualize_analysis(converted_file, music_results)
                print("传统可视化图表已生成")
            except Exception as viz_error:
                print(f"传统可视化失败: {viz_error}")

            # 3. 生成本地装饰画
            try:
                # 确保输出目录存在
                output_dir = "analysis_results"
                os.makedirs(output_dir, exist_ok=True)

                art_path = os.path.join(output_dir, "music_art.png")

                # 检查Matplotlib版本
                matplotlib_version = matplotlib.__version__
                print(f"Matplotlib版本: {matplotlib_version}")

                # 尝试生成装饰画
                print("生成装饰画...")
                art_path = generate_music_art(
                    music_results,
                    save_path=art_path,
                    width=1600,
                    height=1200,
                    dpi=200
                )
                print(f"音乐装饰画已生成: {art_path}")

                # 验证装饰画文件
                if not os.path.exists(art_path):
                    print("警告: 装饰画文件未创建，尝试使用备用路径")
                    alt_path = os.path.join(os.getcwd(), "music_art.png")
                    if os.path.exists(alt_path):
                        art_path = alt_path
                        print(f"使用备用路径: {art_path}")
                    else:
                        print("装饰画创建失败，文件不存在")
            except Exception as art_error:
                print(f"装饰画生成失败: {art_error}")
                # 打印详细错误信息
                import traceback

                print("错误详细信息:")
                traceback.print_exc()
        else:
            print("音乐分析失败，无法生成装饰画")

        print("\n程序执行完成!")

    except FileNotFoundError as fnf_error:
        print(f"文件错误: {fnf_error}")
        print("请确保指定的音频文件存在")
    except Exception as main_error:
        print(f"主程序发生错误: {main_error}")
        import traceback

        traceback.print_exc()

    # 在Windows上保持窗口打开
    if os.name == 'nt':
        input("按Enter键退出程序...")