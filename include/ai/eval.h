/*
 * blupig
 * Copyright (C) 2016-2017 Yunzhu Li
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * any later version.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.

 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef INCLUDE_AI_EVAL_H_
#define INCLUDE_AI_EVAL_H_

#define kRenjuAiEvalWinningScore 10000
#define kRenjuAiEvalThreateningScore 300

class RenjuAIEval {
 public:
    RenjuAIEval();
    ~RenjuAIEval();

    // 评估这个游戏状态的得分
    static int evalState(const char *gs, int player);

    // 评估某个下法的得分
    static int evalMove(const char *gs, int r, int c, int player);

    // 检查是否有棋手获胜
    static int winningPlayer(const char *gs);

// Allow testing private members in this class
#ifndef BLUPIG_TEST
 private:
#endif
    // 一个方向的“局势”
    struct DirectionMeasurement {
        char length;          // 这个方向的“局势”的长度
        char block_count;     // 这个方向的“局势”两段被其他棋子或边界堵住的数量（0-2个）
        char space_count;     // 这个方向的“局势”可以空出的棋子的数量，也就是允许不连续的下棋
    };

    // 一个方向的“局势”的棋谱
    struct DirectionPattern {
        char min_occurrence;  // 最小匹配次数
        char length;          // 这个方向的“局势”的长度
        char block_count;     // 这个方向的“局势”两段被其他棋子或边界堵住的数量（0-2个）
        char space_count;     // 这个方向的“局势”可以空出的棋子的数量，也就是允许不连续的下棋
    };

    // 用来存储生成的棋谱
    static DirectionPattern *preset_patterns;

    // 用于保存每个棋谱的得分
    static int *preset_scores;

    // 在内存中生成棋谱
    static void generatePresetPatterns(DirectionPattern **preset_patterns,
                                       int **preset_scores,
                                       int *preset_patterns_size,
                                       int *preset_patterns_skip);

    // 评估四个方向的局势得分
    static int evalADM(DirectionMeasurement *all_direction_measurement);

    // 尝试匹配某个方向的局势和棋谱
    static int matchPattern(DirectionMeasurement *all_direction_measurement,
                            DirectionPattern *patterns);

    // 测量四个方向的局势
    static void measureAllDirections(const char *gs,
                                     int r,
                                     int c,
                                     int player,
                                     bool consecutive,
                                     RenjuAIEval::DirectionMeasurement *adm);

    // 测量单个方向的局势
    static void measureDirection(const char *gs,
                                 int r, int c,
                                 int dr, int dc,
                                 int player,
                                 bool consecutive,
                                 RenjuAIEval::DirectionMeasurement *result);
};

#endif  // INCLUDE_AI_EVAL_H_
