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

#include <ai/negamax.h>
#include <ai/eval.h>
#include <ai/utils.h>
#include <utils/globals.h>
#include <algorithm>
#include <climits>
#include <cstring>
#include <cstdlib>
#include <ctime>
#include <iostream>

// 对于不同搜索深度，本层允许的宽度不一样。
// 对于较浅的层级，搜索宽度较大，反之较小
int RenjuAINegamax::presetSearchBreadth[5] = {17, 7, 5, 3, 3};

// 迭代加深时评估分支数，用于预估搜索时间
#define kAvgBranchingFactor 3

// 最大搜索深度
#define kMaximumDepth 16

// 定义每层的分数的“衰减比例”，详情请看调用了此define的代码
#define kScoreDecayFactor 0.95f

// 提供给外部调用的启发式Nagamax算法
// 
// 参数：
// gs：游戏状态，即当前下了子的棋盘。可以看作是行优先存储的棋盘
// player：程序的使用的棋子颜色
// depth：搜索深度
// time_limit：搜索时间限制
// enable_ab_pruning：是否启用alpha-beta剪枝
// actual_depth：回传实际的搜索深度
// move_r：计算出的下一步应下棋子的行
// move_c：计算出的下一步应下的棋子的列
void RenjuAINegamax::heuristicNegamax(const char *gs, int player, int depth, int time_limit, bool enable_ab_pruning,
                                      int *actual_depth, int *move_r, int *move_c) {
    // Check arguments
    if (gs == nullptr ||
        player < 1 || player > 2 ||
        depth == 0 || depth < -1 ||
        time_limit < 0) return;

    //备份当前游戏状态，即棋盘，因为每次调用另一签名的heuristicNegamax方法都会改写_gs
    char *_gs = new char[g_gs_size];
    memcpy(_gs, gs, g_gs_size);

    // 程序默认是使用迭代加深的搜索策略，但如果棋局刚开始，
    // 可以直接设置一个深度进行搜索以加快速度，这里深度为6
    int _cnt = 0;
    for (int i = 0; i < static_cast<int>(g_gs_size); i++)
        if (_gs[i] != 0) _cnt++;

    if (_cnt <= 2) depth = 6;

    //根据逐层调用发现，depth传入时是-1，
    //意味着如果depth是-1，即使用迭代加深的搜索策略
    //否则搜索到指定深度即停止，且搜索只发生一次
    if (depth > 0) {
        //设置回传的实际搜索深度
        if (actual_depth != nullptr) *actual_depth = depth;
        //调用核心算法计算下棋位置
        heuristicNegamax(_gs, player, depth, depth, enable_ab_pruning,
                         INT_MIN / 2, INT_MAX / 2, move_r, move_c);
    } else {

        std::clock_t c_start = std::clock();
        //使用迭代加深的搜索策略，直到搜索时间超过了预设的time_limit，
        //或搜索深度超过上限kMaximumDepth
        for (int d = 6;; d += 2) {
            std::clock_t c_iteration_start = std::clock();

            //搜索前还原上次迭代加深搜索修改的棋局
            memcpy(_gs, gs, g_gs_size);

            //以本次迭代深度d进行启发式Negamax搜索
            heuristicNegamax(_gs, player, d, d, enable_ab_pruning,
                             INT_MIN / 2, INT_MAX / 2, move_r, move_c);

            //用于计算是否超时
            std::clock_t c_iteration = (std::clock() - c_iteration_start) * 1000 / CLOCKS_PER_SEC;
            std::clock_t c_elapsed = (std::clock() - c_start) * 1000 / CLOCKS_PER_SEC;

            //如果搜索时间超过了限制或搜索深度超过了限制则退出
            if (c_elapsed + (c_iteration * kAvgBranchingFactor * kAvgBranchingFactor) > time_limit ||
                d >= kMaximumDepth) {
                if (actual_depth != nullptr) *actual_depth = d;
                break;
            }
        }
    }
    delete[] _gs;
}



// 核心算法，用于进行搜索。该方法递归调用，传入指定的搜索深度
// 
// 参数：
// 
// gs：游戏状态
// player：程序使用的棋子颜色
// initial_depth：初始深度
// depth：本次调用的深度
// enable_ab_pruning：是否进行alpha-beta剪枝
// alpha：alpha的值
// beta：beta的值
// move_r：计算出的下一步应下棋子的行
// move_c：计算出的下一步应下的棋子的列
int RenjuAINegamax::heuristicNegamax(char *gs, int player, int initial_depth, int depth,
                                     bool enable_ab_pruning, int alpha, int beta,
                                     int *move_r, int *move_c) {
    // 全局生成结点数目增1
    ++g_node_count;

    // 保存当前最高分数的走法
    int max_score = INT_MIN;
    // opponent是玩家
    int opponent = player == 1 ? 2 : 1;

    // 针对AI和玩家生成所有可走的位置，并按位置的启发值排序
    // candidate_moves的走法进行深度搜索，所以candidate_moves就是当前深度的可扩展结点
    std::vector<Move> moves_player, moves_opponent, candidate_moves;
    searchMovesOrdered(gs, player, &moves_player);
    searchMovesOrdered(gs, opponent, &moves_opponent);

    // 如果AI无棋可走则退出
    if (moves_player.size() == 0) return 0;

    // 如果AI只有一个位置可走，或者有“绝招”可走，就走这一步然后直接退出
    // 绝招是指启发值超过指定限制的，即超过kRenjuAiEvalWinningScore，下绝招位置胜率会大大提升
    if (moves_player.size() == 1 || moves_player[0].heuristic_val >= kRenjuAiEvalWinningScore) {
        auto move = moves_player[0];
        if (move_r != nullptr) *move_r = move.r;
        if (move_c != nullptr) *move_c = move.c;
        return move.heuristic_val;
    }

    // 如果对方能出“绝招”，堵住对方出绝招的位置

    // 用于标记是否该步是用来堵绝招
    bool block_opponent = false;
    int tmp_size = std::min(static_cast<int>(moves_opponent.size()), 2);
    if (moves_opponent[0].heuristic_val >= kRenjuAiEvalThreateningScore) {
        block_opponent = true;
        for (int i = 0; i < tmp_size; ++i) {
            auto move = moves_opponent[i];

            // 堵住绝招后重新评估该步的启发值
            move.heuristic_val = RenjuAIEval::evalMove(gs, move.r, move.c, player);

            // 将“堵绝招”走法加入候选走法之一
            candidate_moves.push_back(move);
        }
    }

    // 根据深度设置搜索宽度，成负相关
    // 随着本方法被递归调用，搜索宽度会逐渐增大
    int breadth = (initial_depth >> 1) - ((depth + 1) >> 1);
    if (breadth > 4) breadth = presetSearchBreadth[4];
    else             breadth = presetSearchBreadth[breadth];

    // 按照breadth设定的值，添加breadth个启发值最大的走法到候选走法内（即要进行深度搜索的走法）
    tmp_size = std::min(static_cast<int>(moves_player.size()), breadth);
    for (int i = 0; i < tmp_size; ++i)
        candidate_moves.push_back(moves_player[i]);

      // Print heuristic values for debugging
//    if (depth >= 8) {
//        for (int i = 0; i < moves_player.size(); ++i) {
//            auto move = moves_player[i];
//            std::cout << depth << " | " << move.r << ", " << move.c << ": " << move.heuristic_val << std::endl;
//        }
//    }

    // 对每个走法再进行启发式Negamax搜索
    int size = static_cast<int>(candidate_moves.size());
    for (int i = 0; i < size; ++i) {
        auto move = candidate_moves[i];

        // 尝试下棋，修改棋盘状态
        RenjuAIUtils::setCell(gs, move.r, move.c, static_cast<char>(player));

        // 递归调用启发式Negamax算法进行深度搜索
        int score = 0;
        if (depth > 1) score = heuristicNegamax(gs,                 // 游戏状态
                                                opponent,           // 更换下棋的人，由对方下棋，即更换max和min方
                                                initial_depth,      // 最初设定的深度
                                                depth - 1,          // 当前深度
                                                enable_ab_pruning,  // Alpha-Beta剪枝
                                                -beta,              // 交换max和min的分数，对于极大极小值算法而言，层与层之间搜索的敌我双方不同，因此要交换双方的极大极小值
                                                -alpha + move.heuristic_val, //
                                                nullptr,            // 对于启发式深度搜索而言，不需要具体策略
                                                nullptr);           // 只需要得到本层不同结点出发的深度搜索能达到的最优值就可以了

        // 对于每一层来说，下层搜索的分数会以kScoreDecayFactor比例衰减，
        // 即，对于较深层结点得到的分数，会按层级衰减若干倍，因此层级较浅的结点分数更重要，
        // 这是为了让程序选择更快（更浅结点的策略）的走法，避免节外生枝。
        // （有绝招干嘛不先出？留着煲汤？）
        if (score >= 2) score = static_cast<int>(score * kScoreDecayFactor);

        // 计算本步在本层实际得分，由预估的启发值减去下层的值得到
        move.actual_score = move.heuristic_val - score;

        // 设置到候选走法（本层结点）中
        candidate_moves[i].actual_score = move.actual_score;

        // Print actual scores for debugging
//        if (depth >= 8)
//            std::cout << depth << " | " << move.r << ", " << move.c << ": " << move.actual_score << std::endl;

        // 恢复棋盘到搜索前的状态
        RenjuAIUtils::setCell(gs, move.r, move.c, 0);

        // 更新本层宽度搜索得分最大值，试图寻找最大值
        if (move.actual_score > max_score) {
            max_score = move.actual_score;
            if (move_r != nullptr) *move_r = move.r;
            if (move_c != nullptr) *move_c = move.c;
        }

        int max_score_decayed = max_score;
        if (max_score >= 2) max_score_decayed = static_cast<int>(max_score_decayed * kScoreDecayFactor);
        if (max_score > alpha) alpha = max_score;

        // 剪枝
        if (enable_ab_pruning && max_score_decayed >= beta) break;
    }

    // 如果本层是最浅层，就要考虑是否堵住对方的“绝招”
    if (depth == initial_depth && block_opponent && max_score < 0) {
        auto blocking_move = candidate_moves[0];
        int b_score = blocking_move.actual_score;
        if (b_score == 0) b_score = 1;
        if ((max_score - b_score) / static_cast<float>(std::abs(b_score)) < 0.2) {
            if (move_r != nullptr) *move_r = blocking_move.r;
            if (move_c != nullptr) *move_c = blocking_move.c;
            max_score = blocking_move.actual_score;
        }
    }
    return max_score;
}

// 这个函数会尝试在棋盘上所有可以下的位置都放置一个棋子，然后评估每个棋子的启发值。
// 在具体实现时，为了避免搜索范围过大，会将搜索区域收缩到当前已经放置了棋子的矩形区域附近
void RenjuAINegamax::searchMovesOrdered(const char *gs, int player, std::vector<Move> *result) {
    // 清除结果
    result->clear();

    //这个过程就是收缩搜索区域，会生成一个矩形区域，我们称之为“含子区域”
    int min_r = INT_MAX, min_c = INT_MAX, max_r = INT_MIN, max_c = INT_MIN;
    for (int r = 0; r < g_board_size; ++r) {
        for (int c = 0; c < g_board_size; ++c) {
            if (gs[g_board_size * r + c] != 0) {
                if (r < min_r) min_r = r;
                if (c < min_c) min_c = c;
                if (r > max_r) max_r = r;
                if (c > max_c) max_c = c;
            }
        }
    }

    // 因为“含子区域”是紧贴当前已含棋子区域的，但“尝试放置”的区域会比“含子区域”稍宽（看下面的代码）
    // 所以为了避免下面“尝试放置”时放出了搜索范围，会提前再次收缩“含子区域”，
    // 由此得到的区域我们称之为“兼容的尝试放置区域”
    if (min_r - 2 < 0) min_r = 2;
    if (min_c - 2 < 0) min_c = 2;
    if (max_r + 2 >= g_board_size) max_r = g_board_size - 3;
    if (max_c + 2 >= g_board_size) max_c = g_board_size - 3;


    // 搜索整个“尝试放置区域”，这个范围是由“兼容的尝试放置区域”每边向外扩展两格得到的
    for (int r = min_r - 2; r <= max_r + 2; ++r) {
        for (int c = min_c - 2; c <= max_c + 2; ++c) {

            // 已经下了棋子的区域就不尝试放置了
            if (gs[g_board_size * r + c] != 0) continue;

            // 如果这个位置是一个远离当前“棋子团”的下棋位置，就不评估它了，直接跳过。
            // 也就是说程序不会无端地把棋子下在远离棋子集中区域的地方
            if (RenjuAIUtils::remoteCell(gs, r, c)) continue;

            Move m;
            m.r = r;
            m.c = c;

            // 调用启发式评估函数评估这个走法的启发值
            m.heuristic_val = RenjuAIEval::evalMove(gs, r, c, player);

            // 添加走法
            result->push_back(m);
        }
    }
    //按启发值从大到小排序（通过Move类型重载的<运算符进行）
    std::sort(result->begin(), result->end());
}

// 这个方法在整个项目中没有调用
int RenjuAINegamax::negamax(char *gs, int player, int depth, int *move_r, int *move_c) {
    // Initialize with a minimum score
    int max_score = INT_MIN;

    // Eval game state
    if (depth == 0) return RenjuAIEval::evalState(gs, player);

    // Loop through all cells
    for (int r = 0; r < g_board_size; ++r) {
        for (int c = 0; c < g_board_size; ++c) {
            // Consider only empty cells
            if (RenjuAIUtils::getCell(gs, r, c) != 0) continue;

            // Skip remote cells (no pieces within 2 cells)
            if (RenjuAIUtils::remoteCell(gs, r, c)) continue;

            // Execute move
            RenjuAIUtils::setCell(gs, r, c, static_cast<char>(player));

            // Run negamax recursively
            int s = -negamax(gs,                   // Game state
                             player == 1 ? 2 : 1,  // Change player
                             depth - 1,            // Reduce depth by 1
                             nullptr,              // Result move not required
                             nullptr);

            // Restore
            RenjuAIUtils::setCell(gs, r, c, 0);

            // Update max score
            if (s > max_score) {
                max_score = s;
                if (move_r != nullptr) *move_r = r;
                if (move_c != nullptr) *move_c = c;
            }
        }
    }
    return max_score;
}
