"""
GoalFy Learning Experience Agent demo
一个智能学习和推荐系统，基于rich_expert_validation.json经验库
"""

import os
import sys
import json
import time
import argparse
from typing import Dict, List, Optional
from experienceagent.controller_agent import ControllerAgent

class GoalFyLearningAgent:
    """GoalFy学习智能体，提供交互式经验学习和推荐"""
    
    def __init__(self, db_path: str = "rich_expert_validation.json"):
        """
        初始化GoalFy学习智能体
        
        Args:
            db_path: 经验库路径
        """
        print("初始化GoalFy学习智能体...")
        self.controller = ControllerAgent(db_path)
        self.current_task = None
        
        # 确认经验库存在
        if not os.path.exists(db_path):
            print(f"警告: 经验库文件不存在 ({db_path})，将创建新的经验库")
            # 创建空的经验库文件
            with open(db_path, "w", encoding="utf-8") as f:
                json.dump({}, f)
        
        print(f"已加载经验库: {db_path}")
        
        # 生成shuchu.json的默认占位符
        if not os.path.exists("shuchu.json"):
            placeholder = {
                "task": "初始化任务",
                "version": 1,
                "trust_score": 0.5,
                "fragments": [],
                "workflow_plan": {"steps": []}
            }
            with open("shuchu.json", "w", encoding="utf-8") as f:
                json.dump(placeholder, f, ensure_ascii=False, indent=2)
    
    def start_session(self, task_name: str = None) -> None:
        """启动新会话"""
        if not task_name:
            task_name = input("请输入任务名称 (直接回车使用默认名称): ")
            if not task_name:
                task_name = f"学习任务 {time.strftime('%Y-%m-%d %H:%M')}"
        
        result = self.controller.new_session(task_name)
        self.current_task = task_name
        print(f"\n{result['message']}")
        print(f"会话ID: {result['session_id']}")
    
    def add_why_fragment(self) -> None:
        """添加WHY片段"""
        print("\n=== 添加WHY片段 ===")
        print("请提供以下信息:")
        
        goal = input("目标 (goal): ")
        background = input("背景 (background): ")
        
        print("约束条件 (constraints, 多个条件以空行结束):")
        constraints = []
        while True:
            constraint = input(f"约束条件 #{len(constraints)+1} (空行结束): ")
            if not constraint:
                break
            constraints.append(constraint)
        
        expected_outcome = input("预期效果 (expected_outcome): ")
        
        # 创建片段数据
        fragment_data = {
            "goal": goal,
            "background": background,
            "constraints": constraints,
            "expected_outcome": expected_outcome
        }
        
        # 添加片段
        result = self.controller.add_fragment("WHY", fragment_data)
        print(f"\n{result['message']}")
        
        # 保存到shuchu.json
        self.controller.save_to_file()
    
    def add_how_fragment(self) -> None:
        """添加HOW片段"""
        print("\n=== 添加HOW片段 ===")
        print("请提供执行步骤 (多个步骤以空行结束):")
        
        steps = []
        while True:
            print(f"\n步骤 #{len(steps)+1}:")
            page = input("页面 (page): ")
            if not page:
                break
                
            action = input("动作 (action): ")
            element = input("元素 (element): ")
            intent = input("意图 (intent): ")
            
            step = {
                "page": page,
                "action": action,
                "element": element,
                "intent": intent
            }
            steps.append(step)
            
            more = input("继续添加步骤? (y/n): ").lower()
            if more != 'y':
                break
        
        # 创建片段数据
        fragment_data = {
            "steps": steps
        }
        
        # 添加片段
        result = self.controller.add_fragment("HOW", fragment_data)
        print(f"\n{result['message']}")
        
        # 更新工作流计划
        workflow_steps = [f"{step['action']} {step['element']}" for step in steps]
        self.controller.update_workflow_plan(workflow_steps)
        
        # 保存到shuchu.json
        self.controller.save_to_file()
    
    def add_check_fragment(self) -> None:
        """添加CHECK片段"""
        print("\n=== 添加CHECK片段 ===")
        print("请提供验证规则 (多个规则以空行结束):")
        
        rules = []
        while True:
            rule = input(f"规则 #{len(rules)+1} (空行结束): ")
            if not rule:
                break
            rules.append(rule)
        
        # 创建片段数据
        fragment_data = {
            "rules": rules
        }
        
        # 添加片段
        result = self.controller.add_fragment("CHECK", fragment_data)
        print(f"\n{result['message']}")
        
        # 保存到shuchu.json
        self.controller.save_to_file()
    
    def get_recommendations(self) -> None:
        """获取经验推荐"""
        print("\n=== 获取经验推荐 ===")
        task_description = input("请描述您的任务需求: ")
        if not task_description:
            task_description = self.current_task
        
        result = self.controller.recommend_fragments(task_description)
        
        print("\n推荐结果:")
        if result["has_recommendations"]:
            for frag_type, items in result["recommendations"].items():
                print(f"\n{frag_type}类型推荐 ({len(items)}个):")
                for i, item in enumerate(items[:3]):  # 只显示前3个
                    print(f"{i+1}. 来源: {item['task_name']}")
                    print(f"   相似度: {item['similarity']:.2f}")
                    
                    # 根据片段类型显示关键信息
                    if frag_type == "WHY":
                        print(f"   目标: {item['content'].get('goal', '')}")
                    elif frag_type == "HOW" and "steps" in item['content']:
                        steps_count = len(item['content']['steps'])
                        print(f"   步骤数: {steps_count}")
                        if steps_count > 0:
                            print(f"   第一步: {item['content']['steps'][0].get('action', '')} {item['content']['steps'][0].get('element', '')}")
                    elif frag_type == "CHECK" and "rules" in item['content']:
                        rules_count = len(item['content']['rules'])
                        print(f"   规则数: {rules_count}")
                
                # 询问是否要采用推荐
                choice = input("\n是否要采用这些推荐? (y/n): ").lower()
                if choice == 'y':
                    for item in items[:1]:  # 采用第一个推荐
                        self.controller.add_fragment(frag_type, item['content'])
                        print(f"已添加{frag_type}类型片段")
                        # 保存到shuchu.json
                        self.controller.save_to_file()
        else:
            print("没有找到相关推荐。")
    
    def enhance_experience(self) -> None:
        """增强当前经验"""
        print("\n=== 增强当前经验 ===")
        
        result = self.controller.enhance_experience()
        
        print(f"质量评级: {result['quality_level']}")
        
        if result["has_enhancement_potential"]:
            print("\n增强建议:")
            for suggestion in result["suggestions"]:
                print(f"- {suggestion}")
            
            if result["missing_types"]:
                print(f"\n缺少的片段类型: {', '.join(result['missing_types'])}")
                
                for missing_type in result["missing_types"]:
                    if missing_type in result["complementary_recommendations"]:
                        items = result["complementary_recommendations"][missing_type]
                        print(f"\n推荐的{missing_type}片段 ({len(items)}个):")
                        for i, item in enumerate(items[:2]):  # 只显示前2个
                            print(f"{i+1}. 来源: {item['task_name']}")
                            print(f"   相似度: {item['similarity']:.2f}")
                        
                        # 询问是否要添加
                        choice = input(f"\n是否要添加推荐的{missing_type}片段? (y/n): ").lower()
                        if choice == 'y' and items:
                            self.controller.add_fragment(missing_type, items[0]['content'])
                            print(f"已添加{missing_type}类型片段")
                            # 保存到shuchu.json
                            self.controller.save_to_file()
        else:
            print("当前经验质量良好，无需增强。")
    
    def process_query(self) -> None:
        """处理用户查询"""
        print("\n=== 处理用户查询 ===")
        query = input("请输入您的问题: ")
        
        if not query:
            print("查询为空，已取消。")
            return
            
        result = self.controller.process_user_input(query)
        
        print(f"\n系统回复: {result['message']}")
        
        if result["similar_experiences"]:
            print("\n找到的相关经验:")
            for i, exp in enumerate(result["similar_experiences"]):
                print(f"{i+1}. {exp['task_name']} (相似度: {exp['similarity']:.2f})")
                print(f"   原因: {exp['reason']}")
    
    def save_experience(self) -> None:
        """保存当前经验"""
        print("\n=== 保存当前经验 ===")
        
        task_name = input(f"请确认任务名称 [{self.current_task}]: ")
        if task_name:
            self.current_task = task_name
        
        # 保存到数据库和shuchu.json
        result = self.controller.save_current_experience(self.current_task)
        self.controller.save_to_file()
        
        print(f"\n{result['message']}")
        print(f"已包含 {result['fragment_count']} 个片段")
    
    def view_current_experience(self) -> None:
        """查看当前经验"""
        if not self.controller.current_experience:
            print("\n当前没有活动的经验。")
            return
            
        export_result = self.controller.export_experience()
        print("\n当前经验:")
        print(export_result["data"])
    
    def run(self) -> None:
        """运行交互式学习智能体"""
        print("\n====== GoalFy Learning Experience Agent ======")
        print("欢迎使用GoalFy学习体验智能体!")
        
        # 启动会话
        self.start_session()
        
        while True:
            print("\n======== 主菜单 ========")
            print(f"当前任务: {self.current_task}")
            print("1. 添加WHY片段 (目标和背景)")
            print("2. 添加HOW片段 (步骤和行为)")
            print("3. 添加CHECK片段 (验证规则)")
            print("4. 获取经验推荐")
            print("5. 增强当前经验")
            print("6. 处理用户查询")
            print("7. 查看当前经验")
            print("8. 保存当前经验")
            print("9. 开始新会话")
            print("0. 退出")
            
            choice = input("\n请选择操作: ")
            
            if choice == "1":
                self.add_why_fragment()
            elif choice == "2":
                self.add_how_fragment()
            elif choice == "3":
                self.add_check_fragment()
            elif choice == "4":
                self.get_recommendations()
            elif choice == "5":
                self.enhance_experience()
            elif choice == "6":
                self.process_query()
            elif choice == "7":
                self.view_current_experience()
            elif choice == "8":
                self.save_experience()
            elif choice == "9":
                self.start_session()
            elif choice == "0":
                # 退出前保存会话
                save_session = input("是否保存当前会话? (y/n): ").lower()
                if save_session == 'y':
                    self.controller.save_session()
                    self.controller.save_to_file()
                print("感谢使用GoalFy学习智能体，再见!")
                break
            else:
                print("无效的选择，请重试。")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="GoalFy Learning Experience Agent")
    parser.add_argument("--db", default="rich_expert_validation.json", help="经验库路径")
    parser.add_argument("--task", help="初始任务名称")
    
    args = parser.parse_args()
    
    try:
        # 初始化并运行智能体
        agent = GoalFyLearningAgent(args.db)
        agent.run()
    except KeyboardInterrupt:
        print("\n程序已被用户中断。")
        sys.exit(0)
    except Exception as e:
        print(f"\n发生错误: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    # 运行主函数
    main()