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

#include <ai/eval.h>
#include <ai/utils.h>
#include <utils/globals.h>
#include <stdlib.h>
#include <algorithm>
#include <climits>
#include <cstring>

// Initialize global variables
RenjuAIEval::DirectionPattern *RenjuAIEval::preset_patterns = nullptr;
int *RenjuAIEval::preset_scores = nullptr;
int preset_patterns_size = 0;
int preset_patterns_skip[6] = {0};

int RenjuAIEval::evalState(const char *gs, int player) {
    // Check parameters
    if (gs == nullptr ||
        player < 1 || player > 2) return 0;

    // 随意下，评估每一步
    int score = 0;
    for (int r = 0; r < g_board_size; ++r) {
        for (int c = 0; c < g_board_size; ++c) {
            score += evalMove(gs, r, c, player);
        }
    }
    return score;
}

// 用于在启发式Negamax算法中评估启发值
int RenjuAIEval::evalMove(const char *gs, int r, int c, int player) {
    // Check parameters
    if (gs == nullptr ||
        player < 1 || player > 2) return 0;

    // 全局评估次数增1
    ++g_eval_count;

    // 生成“棋谱”，下面会按棋谱招数计算得分
    if (preset_patterns == nullptr) {
        generatePresetPatterns(&preset_patterns, &preset_scores, &preset_patterns_size, preset_patterns_skip);
    }

    // 对于某个下法，测量它8个方向上棋子的分布情况，可以认为是8个方向的“局势”
    DirectionMeasurement adm[4];

    // 连续和不连续的统计“局势”
    int max_score = 0;
    for (bool consecutive = false;; consecutive = true) {
        // 测量所有方向的“局势”
        measureAllDirections(gs, r, c, player, consecutive, adm);

        // 统计出了棋子分布情况（局势），通过不同方向的分布计算出不同的分数
        int score = evalADM(adm);

        // Prefer consecutive
        // if (!consecutive) score *= 0.9;

        // 不要求方向上己方棋子连续
        max_score = std::max(max_score, score);

        if (consecutive) break;
    }
    return max_score;
}

// 通过某个下法测量出的各个方向的情况（“局势”），计算出分数
int RenjuAIEval::evalADM(DirectionMeasurement *all_direction_measurement) {
    int score = 0;
    int size = preset_patterns_size;

    // 每个方向评估出来的分数相加
    // 每个方向棋子越长，分数越高，因为五子棋越长越好
    // “棋谱”（present_patterns）内的下法有的长度超过了当前方向连续棋子的长度，
    // 因此把这些棋谱忽略掉
    int max_measured_len = 0;
    for (int i = 0; i < 4; i++) {
        int len = all_direction_measurement[i].length;
        max_measured_len = len > max_measured_len ? len : max_measured_len;
        score += len - 1;
    }
    int start_pattern = preset_patterns_skip[max_measured_len];

    // 将所有方向的“局势”与“棋谱”进行匹配，如果匹配到“棋谱”，按照棋谱的分数给分
    for (int i = start_pattern; i < size; ++i) {
        score += matchPattern(all_direction_measurement, &preset_patterns[2 * i]) * preset_scores[i];

        // 如果匹配到了“绝招”棋谱，直接退出，节省时间
        if (score >= kRenjuAiEvalThreateningScore) break;
    }

    return score;
}

// 将各个方向的“局势”与“棋谱”进行匹配
int RenjuAIEval::matchPattern(DirectionMeasurement *all_direction_measurement,
                              DirectionPattern *patterns) {
    // Check arguments
    if (all_direction_measurement == nullptr) return -1;
    if (patterns == nullptr) return -1;

    // 全局查找“棋谱”次数增1
    g_pm_count++;

    // Initialize match_count to INT_MAX since minimum value will be output
    int match_count = INT_MAX, single_pattern_match = 0;

    // 每个方向的局势只查找两个棋谱
    for (int i = 0; i < 2; ++i) {
        auto p = patterns[i];
        if (p.length == 0) break;

        // Initialize counter
        single_pattern_match = 0;

        // 查找4个方向的局势
        for (int j = 0; j < 4; ++j) {
            auto dm = all_direction_measurement[j];

            // 如果棋谱和局势完全匹配，匹配的数量增加
            // 注意有的棋谱不要求block和space
            if (dm.length == p.length &&
                (p.block_count == -1 || dm.block_count == p.block_count) &&
                (p.space_count == -1 || dm.space_count == p.space_count)) {
                single_pattern_match++;
            }
        }

        // Consider minimum number of occurrences
        single_pattern_match /= p.min_occurrence;

        // 取匹配到的局势数量最少的棋谱的匹配次数
        match_count = match_count >= single_pattern_match ? single_pattern_match : match_count;
    }
    return match_count;
}

void RenjuAIEval::measureAllDirections(const char *gs,
                                       int r,
                                       int c,
                                       int player,
                                       bool consecutive,
                                       RenjuAIEval::DirectionMeasurement *adm) {
    // Check arguments
    if (gs == nullptr) return;
    if (r < 0 || r >= g_board_size || c < 0 || c >= g_board_size) return;

    // Measure 4 directions
    measureDirection(gs, r, c, 0,  1, player, consecutive, &adm[0]);
    measureDirection(gs, r, c, 1,  1, player, consecutive, &adm[1]);
    measureDirection(gs, r, c, 1,  0, player, consecutive, &adm[2]);
    measureDirection(gs, r, c, 1, -1, player, consecutive, &adm[3]);
}

// 测量某个方向的棋局局势并通过result返回，
// 一个方向的局势大概是某个方向己方棋子一条连起来的情况，
// 如果consecutive为true则必须是连续的己方棋子，否则可以空一格，但不能是对方棋子。
// 本方法就是尝试向某个方向延伸，看指定方向“一条”的棋子的数量，以及检测有没有对方棋子堵在“一条”的两端
void RenjuAIEval::measureDirection(const char *gs,
                                   int r, int c,
                                   int dr, int dc,
                                   int player,
                                   bool consecutive,
                                   RenjuAIEval::DirectionMeasurement *result) {
    // Check arguments
    if (gs == nullptr) return;
    if (r < 0 || r >= g_board_size || c < 0 || c >= g_board_size) return;
    if (dr == 0 && dc == 0) return;

    // Initialization
    int cr = r, cc = c;
    result->length = 1, result->block_count = 2, result->space_count = 0;

    int space_allowance = 1;
    if (consecutive) space_allowance = 0;

    for (bool reversed = false;; reversed = true) {
        while (true) {
            // 尝试向某个方向延伸一格
            cr += dr; cc += dc;

            // 检测方向是否有效
            if (cr < 0 || cr >= g_board_size || cc < 0 || cc >= g_board_size) break;

            // 获取延伸的格子的下棋情况
            int cell = gs[g_board_size * cr + cc];

            // 如果延伸的格子没有被下，就看看是否要求连续
            if (cell == 0) {
                // 如果space_allowance大于0，即允许一定数量的空格（不连续），则此次延伸合法
                // 但是允许空格的数量要减1
                if (space_allowance > 0 && RenjuAIUtils::getCell(gs, cr + dr, cc + dc) == player) {
                    space_allowance--; result->space_count++;
                    continue;
                // 如果要求连续，则这个延伸不合法，本次延伸结束
                // 但因为这“一条”尽头没有对方棋子堵住，所以block_count减1
                } else {
                    result->block_count--;
                    break;
                }
            }

            // 如果延伸的格子是对方的棋子，延伸中止
            if (cell != player) break;

            // 这个方向“一条”的长度
            result->length++;
        }

        // 从一开始下棋的位置方向延伸一次
        if (reversed) break;
        cr = r; cc = c;
        dr = -dr; dc = -dc;
    }

    // 如果这“一条”大于5个棋子，
    if (result->length >= 5) {
        if (result->space_count == 0) {
            result->length = 5;
            result->block_count = 0;
        } else {
            result->length = 4;
            result->block_count = 1;
        }
    }
}

void RenjuAIEval::generatePresetPatterns(DirectionPattern **preset_patterns,
                                         int **preset_scores,
                                         int *preset_patterns_size,
                                         int *preset_patterns_skip) {
    const int _size = 11;
    preset_patterns_skip[5] = 0;
    preset_patterns_skip[4] = 1;
    preset_patterns_skip[3] = 7;
    preset_patterns_skip[2] = 10;

    preset_patterns_skip[1] = _size;
    preset_patterns_skip[0] = _size;

    DirectionPattern patterns[_size * 2] = {
        {1, 5,  0,  0}, {0, 0,  0,  0},  // 10000
        {1, 4,  0,  0}, {0, 0,  0,  0},  // 700
        {2, 4,  1,  0}, {0, 0,  0,  0},  // 700
        {2, 4, -1,  1}, {0, 0,  0,  0},  // 700
        {1, 4,  1,  0}, {1, 4, -1,  1},  // 700
        {1, 4,  1,  0}, {1, 3,  0, -1},  // 500
        {1, 4, -1,  1}, {1, 3,  0, -1},  // 500
        {2, 3,  0, -1}, {0, 0,  0,  0},  // 300
        // {1, 4,  1,  0}, {0, 0,  0,  0},  // 1
        // {1, 4, -1,  1}, {0, 0,  0,  0},  // 1
        {3, 2,  0, -1}, {0, 0,  0,  0},  // 50
        {1, 3,  0, -1}, {0, 0,  0,  0},  // 20
        {1, 2,  0, -1}, {0, 0,  0,  0}   // 9
    };

    int scores[_size] = {
        10000,
        700,
        700,
        700,
        700,
        500,
        500,
        300,
        // 1,
        // 1,
        50,
        20,
        9
    };

    *preset_patterns = new DirectionPattern[_size * 2];
    *preset_scores   = new int[_size];

    memcpy(*preset_patterns, patterns, sizeof(DirectionPattern) * _size * 2);
    memcpy(*preset_scores, scores, sizeof(int) * _size);

    *preset_patterns_size = _size;
}

int RenjuAIEval::winningPlayer(const char *gs) {
    if (gs == nullptr) return 0;
    for (int r = 0; r < g_board_size; ++r) {
        for (int c = 0; c < g_board_size; ++c) {
            int cell = gs[g_board_size * r + c];
            if (cell == 0) continue;
            for (int dr = -1; dr <= 1; ++dr) {
                for (int dc = -1; dc <= 1; ++dc) {
                    if (dr == 0 && dc <= 0) continue;
                    DirectionMeasurement dm;
                    measureDirection(gs, r, c, dr, dc, cell, 1, &dm);
                    if (dm.length >= 5) return cell;
                }
            }
        }
    }
    return 0;
}
