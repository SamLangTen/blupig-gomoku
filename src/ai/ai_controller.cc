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

#include <ai/ai_controller.h>
#include <ai/eval.h>
#include <ai/negamax.h>
#include <ai/utils.h>
#include <utils/globals.h>
#include <cstring>

// 暴露出用于外部调用的方法，调用本目录下的其他代码产生下一步的下法
void RenjuAIController::generateMove(const char *gs, int player, int search_depth, int time_limit,
                           int *actual_depth, int *move_r, int *move_c, int *winning_player,
                           unsigned int *node_count, unsigned int *eval_count, unsigned int *pm_count) {
    // 检查参数
    if (gs == nullptr ||
        player  < 1 || player > 2 ||
        search_depth == 0 || search_depth > 10 ||
        time_limit < 0 ||
        move_r == nullptr || move_c == nullptr) return;

    // 全局计数器，每一步都会统计评估次数和局势棋谱配对次数
    g_eval_count = 0;
    g_pm_count = 0;

    // 初始化数据
    *move_r = -1;
    *move_c = -1;
    int _winning_player = 0;
    if (actual_depth != nullptr) *actual_depth = 0;

    // 检查是否有玩家获胜
    _winning_player = RenjuAIEval::winningPlayer(gs);
    if (_winning_player != 0) {
        if (winning_player != nullptr) *winning_player = _winning_player;
        return;
    }

    // 备份游戏状态
    char *_gs = new char[g_gs_size];
    std::memcpy(_gs, gs, g_gs_size);

    // 运行启发式Negamax算法
    RenjuAINegamax::heuristicNegamax(_gs, player, search_depth, time_limit, true, actual_depth, move_r, move_c);

    // 备份游戏状态，下棋并将走棋方式通过move_r和move_c输出
    std::memcpy(_gs, gs, g_gs_size);
    RenjuAIUtils::setCell(_gs, *move_r, *move_c, static_cast<char>(player));

    // 检查是否有获胜的
    _winning_player = RenjuAIEval::winningPlayer(_gs);

    // 输出
    if (winning_player != nullptr) *winning_player = _winning_player;
    if (node_count != nullptr) *node_count = g_node_count;
    if (eval_count != nullptr) *eval_count = g_eval_count;
    if (pm_count != nullptr) *pm_count = g_pm_count;

    delete[] _gs;
}
