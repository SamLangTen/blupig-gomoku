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

#ifndef INCLUDE_AI_NEGAMAX_H_
#define INCLUDE_AI_NEGAMAX_H_

#include <vector>

class RenjuAINegamax {
 public:
    RenjuAINegamax();
    ~RenjuAINegamax();

    static void heuristicNegamax(const char *gs, int player, int depth, int time_limit, bool enable_ab_pruning,
                                 int *actual_depth, int *move_r, int *move_c);

 private:
    // 每层的搜索宽度
    static int presetSearchBreadth[5];

    // 一个候选下法
    struct Move {
        int r;
        int c;
        int heuristic_val;
        int actual_score;

        // 重载<运算符用于给“下法”类排序
        bool operator<(Move other) const {
            return heuristic_val > other.heuristic_val;
        }
    };

    static int heuristicNegamax(char *gs, int player, int initial_depth, int depth,
                                bool enable_ab_pruning, int alpha, int beta,
                                int *move_r, int *move_c);

    // 搜索所有可以下的位置，即宽度搜索
    static void searchMovesOrdered(const char *gs, int player, std::vector<Move> *result);

    // 未使用
    static int negamax(char *gs, int player, int depth,
                       int *move_r, int *move_c);
};

#endif  // INCLUDE_AI_NEGAMAX_H_
