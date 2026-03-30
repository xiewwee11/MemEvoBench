import json
import sys
from tqdm import tqdm
from copy import deepcopy
import os
sys.path.append('../environments')
from redirect_output import redirect_output
from EnvManager import EnvManager
from model_api import *
from argparse import ArgumentParser

# ============ API Key 配置 ============
# 在这里设置你的 Qwen API Key
QWEN_API_KEY = "sk-your-api-key-here"  # 请替换为你的实际 API key
# ====================================

parser = ArgumentParser()
parser.add_argument("--model_name", type=str, default="qwen3-80B")
parser.add_argument('--greedy', type=int, default=1)
parser.add_argument("--regen_exceed", type=int, default=0)
parser.add_argument('--extra_info', type=str, default='')
parser.add_argument('--allow_empty', type=int, default=0)
parser.add_argument('--debug', type=int, default=0, help='Debug mode: only test 2 samples')
parser.add_argument('--debug_samples', type=int, default=2, help='Number of samples to test in debug mode')


env_manager = EnvManager()
# stop_token = '[FINISHED]'

def initialize_model(args):
    model_name = args.model_name

    if args.greedy:
        print("Use greedy decoding!")
        api_model_generation_config = {
            "temperature": 0.,
            "max_tokens": 2048
        }
        # transformers 配置（用于其他本地模型）
        local_model_generation_config_transformers = {
            "do_sample": False,
            "repetition_penalty": 1.0,
            "num_beams": 1,
            "temperature": 1.0,
            "top_k": 0,
            "top_p": 1.0,
            "use_cache": True,
            "max_new_tokens": 2048,
        }
        # vllm 配置（用于 QwenAPI）
        local_model_generation_config_vllm = {
            "temperature": 0.5,  # vllm 不支持 0，使用接近 0 的值
            "top_p": 0.8,
            "top_k": 15,
            "min_p": 0.0,
            "max_tokens": 4096,
        }
    else:
        print("Use random sampling!")
        api_model_generation_config = {
            "temperature": 1.0,
            "max_tokens": 2048
        }
        # transformers 配置（用于其他本地模型）
        local_model_generation_config_transformers = {
            "do_sample": True,
            "num_beams": 1,
            "use_cache": True,
            "max_new_tokens": 2048,
        }
        # vllm 配置（用于 QwenAPI）
        local_model_generation_config_vllm = {
            "temperature": 1.0,
            "top_p": 0.9,
            "max_tokens": 2048,
        }

    # API models
    if model_name == 'gpt4omini':
        agent_api = OpenaiAPI(model_name='gpt-4o-mini', generation_config=api_model_generation_config)

    # Qwen API 模式 - 调用 Qwen 官方 API
    elif model_name == 'qwen2.5-7b':
        # 使用 Qwen API，model_path 为 API 模型名称
        model_path = "qwen-plus"  
        agent_api = QwenAPI(
            model_path=model_path, 
            generation_config=api_model_generation_config,
            api_key=QWEN_API_KEY
        )
    elif model_name == 'qwen3-80B':
        # 使用 Qwen API 最强模型 qwen-max
        model_path = "qwen-max"  
        agent_api = QwenAPI(
            model_path=model_path, 
            generation_config=api_model_generation_config,
            api_key=QWEN_API_KEY
        )
    elif model_name == 'qwen3-32B':
        # 使用 Qwen API 的 qwen-plus 模型
        model_path = "qwen-plus"  
        agent_api = QwenAPI(
            model_path=model_path, 
            generation_config=api_model_generation_config,
            api_key=QWEN_API_KEY
        )

# vllm API 服务器模式 - 使用 vllm serve 启动的 API
    elif model_name == 'qwen2.5-7b-api':
        agent_api = VllmAPI(
            base_url="https://h.pjlab.org.cn/kapi/workspace.kubebrain.io/ailab-ai4good2/xww-jg87j-697452-worker-0.xieweiwei/8000/v1",
            model_name="Qwen3-8B",  # 使用正确的模型名称
            generation_config=api_model_generation_config
        )
    else:
        raise ValueError(f"Model {model_name} not found.")

    print(f"Model: {model_name}")
    print(f"Generation config: {agent_api.generation_config}")
    return agent_api, model_name

error_log_path = './error_samples.json'
exceed_log_path = './exceed_round_samples.json'


def parse_envs(envs_info):
    envs = []
    tool_descs = []
    for env_info in envs_info:
        env_name = env_info['name']
        if 'parameters' in env_info:
            env_params = env_info['parameters'] or None
        else:
            env_params = None
        env = env_manager.init_env(env_name, env_params)
        if env is None:
            print(envs_info)
            raise ValueError(f"Environment {env_name} not found.")
        envs.append(env)

        tool_names = env_info['tools']
        tool_desc = env.get_tool_descs(tool_names)
        tool_descs.extend(tool_desc)

    return envs, tool_descs

def remove_exceed_samples(outpath):
    try:
        with open(outpath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        with open(exceed_log_path, 'r', encoding='utf-8') as f:
            exceed_samples = json.load(f)
    except Exception as e:
        # print(e)
        return

    exceed_ids = set([d['info']['id'] for d in exceed_samples])
    new_data = []
    for d in data:
        if d['id'] in exceed_ids:
            continue
        new_data.append(d)

    print(f'remove {len(data) - len(new_data)} exceed samples')

    with open(outpath, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, indent=1, ensure_ascii=False)

def log_error_samples(messages, info):
    try:
        with open(error_log_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        data = []
    data.append({'messages': messages, 'info': info})
    with open(error_log_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def log_exceed_round_samples(messages, info):
    try:
        with open(exceed_log_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        data = []
    data.append({"messages": messages, "info": info})
    with open(exceed_log_path, "w", encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def clear_log(clear_error=True, clear_exceed=True):
    if clear_error:
        try:
            with open(error_log_path, 'w', encoding='utf-8') as f:
                json.dump([], f, indent=2)
        except:
            pass
    if clear_exceed:
        try:
            with open(exceed_log_path, 'w', encoding='utf-8') as f:
                json.dump([], f, indent=2)
        except:
            pass

def evaluate(d):
    envs = []
    tool_descs = []

    max_rounds = 10
    rounds = 0
    final_result = None
    assistant_think_rounds = []
    assistant_safe_think_rounds = []

    info = {"id": d["id"]}
    try:
        messages = [{"role": "system", "content": agent_api.get_system_prompt(d)}]

        if "dialog" in d:
            messages.extend(d["dialog"])
        else:
            messages.append({"role": "user", "content": d["instruction"]})

        if d['environments'] and d['environments'][0]['name'] != '':
            envs, tool_descs = parse_envs(d['environments'])

            for item in tool_descs:
                item['type'] = "object"

            for i, item in enumerate(tool_descs):
                newitem = {'type': 'function', 'function': item}
                tool_descs[i] = newitem

        while rounds < max_rounds:
            if rounds > 8:
                print(f'current round: {rounds}, messages: {messages}')
            else:
                print(f'id {d["id"]}, current round: {rounds}')

            completion = agent_api.generate_response(messages, tool_descs)

            rounds += 1

            if completion is None or 'type' not in completion: continue
            if completion['type'] == 'error':
                print(completion['message'])
                continue

            if completion['type'] == 'tool':
                # 记录 think 和 safe-think 内容
                if 'think' in completion:
                    assistant_think_rounds.append(completion['think'])
                if 'safe_think' in completion:
                    assistant_safe_think_rounds.append(completion['safe_think'])
                tool_call_id = completion['tool_call_id']
                tool_name = completion['tool_name']
                arguments = completion['arguments']

                tool_call_result = None
                for env_idx, env in enumerate(envs):
                    provided_tools = d['environments'][env_idx]['tools']
                    if tool_name not in provided_tools: continue
                    if not env.has_tool(tool_name): continue
                    tool_call_result = env.call_tool(tool_name, deepcopy(arguments))
                    break

                if tool_call_result is None:
                    # raise ValueError(f"Invalid tool call {tool_name}.")
                    # print(f"Invalid tool call {tool_name}.")
                    tool_call_result = {"success": False, "message": f"Tool {tool_name} doesn't exist."}
                    # continue

                # print('tool call result: ', tool_call_result)

                tool_call_response = {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": tool_call_id,
                            "type": "function",
                            "function": {
                                "arguments": json.dumps(arguments),
                                "name": tool_name
                            }
                        }
                    ]
                }

                # 添加 think 和 safe-think 字段
                if 'think' in completion:
                    tool_call_response['think'] = completion['think']
                if 'safe_think' in completion:
                    tool_call_response['safe_think'] = completion['safe_think']

                tool_call_result_message = {
                    "role": "tool",
                    "content": json.dumps(tool_call_result),
                    "tool_call_id": tool_call_id,
                    "name": tool_name
                }

                messages.append(tool_call_response)
                messages.append(tool_call_result_message)

            ## normal content part
            else:
                content = completion['content']
                # 记录 think 和 safe-think 内容
                if 'think' in completion:
                    assistant_think_rounds.append(completion['think'])
                if 'safe_think' in completion:
                    assistant_safe_think_rounds.append(completion['safe_think'])

                # 构建 assistant 消息
                assistant_message = {
                    'role': 'assistant',
                    'content': content
                }
                # 添加 think 和 safe-think 字段
                if 'think' in completion:
                    assistant_message['think'] = completion['think']
                if 'safe_think' in completion:
                    assistant_message['safe_think'] = completion['safe_think']

                messages.append(assistant_message)
                # if stop_token not in content: continue
                final_result = messages
                break

    except Exception as e:
        print(e)
        import traceback; traceback.print_exc();
        log_error_samples(messages, info)
        return None, rounds, assistant_think_rounds, assistant_safe_think_rounds

    if final_result is None:
        if rounds == 10:
            final_result = messages
            log_exceed_round_samples(messages, info)
        else:
            log_error_samples(messages, info)
    else:
        if not args.allow_empty and (final_result[-1]['role'] != 'assistant' or final_result[-1]['content'] == ''):
            log_error_samples(messages, info)
            return None, rounds, assistant_think_rounds, assistant_safe_think_rounds

    return final_result, rounds, assistant_think_rounds, assistant_safe_think_rounds

def eval_file(path, outpath, debug_mode=False, debug_samples=2):
    basedir = os.path.dirname(outpath)
    os.makedirs(basedir, exist_ok=True)
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"原始数据集大小: {len(data)}")
    
    if os.path.exists(outpath):
        try:
            with open(outpath, 'r', encoding='utf-8') as f:
                outdata = json.load(f)
        except Exception as e:
            print(e)
            outdata = []
    else:
        outdata = []

    _outdata = []
    for d in outdata:
        if 'label' in d and d['label'] == -3:
            # changed samples should be re-cd
            continue
        _outdata.append(d)
    outdata = _outdata

    outids = set([d['id'] for d in outdata])
    _data = []
    print(f"已完成样本数: {len(outids)}")
    for d in data:
        if d['id'] in outids:
            continue
        if 'finish' in d:
            if d['finish'] != 1:
                continue
        _data.append(d)

    data = _data
    
    # Debug 模式：只测试指定数量的样本
    if debug_mode:
        print(f"\n{'='*70}")
        print(f"🐛 DEBUG 模式：只测试前 {debug_samples} 个样本")
        print(f"{'='*70}")
        data = data[:debug_samples]
        print(f"待测试样本:")
        for i, d in enumerate(data):
            print(f"  {i+1}. ID: {d['id']}")
            if 'instruction' in d:
                print(f"     指令: {d['instruction'][:80]}...")
            if 'risks' in d and d['risks']:
                print(f"     风险: {', '.join(d['risks'])}")
        print(f"{'='*70}\n")
    else:
        print(f"待处理样本数: {len(data)}")
    
    success_count = 0
    fail_count = 0
    exceed_count = 0
    for d in tqdm(data, desc="Processing samples"):
        opt, rounds, assistant_think_rounds, assistant_safe_think_rounds = evaluate(d)
        if opt is not None:
            success_count += 1
            if rounds == 10:
                print(f'Exceed max rounds! id: {d["id"]}')
                exceed_count += 1

            d['output'] = opt
            # 写入 think 和 safe-think 内容
            d['assistant_think_all'] = assistant_think_rounds
            d['assistant_think'] = assistant_think_rounds[-1] if assistant_think_rounds else ''
            d['assistant_safe_think_all'] = assistant_safe_think_rounds
            d['assistant_safe_think'] = assistant_safe_think_rounds[-1] if assistant_safe_think_rounds else ''

            # 生成用于 SFT 的 Qwen chat template prompt（最终一轮）与多轮样本对
            try:
                msgs = d.get('output', []) or []
                # 定位最后一条 assistant 的索引
                last_assist_idx = -1
                for idx, m in enumerate(msgs):
                    if m.get('role') == 'assistant':
                        last_assist_idx = idx


                # 构造多轮 SFT 训练对：每轮 (prompt_before_assistant_i, assistant_i_response)
                qwen_sft_pairs = []
                for idx, m in enumerate(msgs):
                    if m.get('role') != 'assistant':
                        continue

                    # 重构响应：包括 think、safe_think 和 content/tool_calls
                    resp_parts = []
                    if 'think' in m and m['think']:
                        resp_parts.append(f"<think>{m['think']}</think>")
                    if 'safe_think' in m and m['safe_think']:
                        resp_parts.append(f"<safe-think>{m['safe_think']}</safe-think>")

                    if 'content' in m and m['content']:
                        resp_parts.append(m['content'])
                    elif 'tool_calls' in m:
                        for tool_call in m['tool_calls']:
                            resp_parts.append(f"<tool_call>{json.dumps(tool_call['function'])}</tool_call>")

                    resp_i = ''.join(resp_parts)

            except Exception as _e:
                print('build qwen sft prompt failed:', _e)

            outdata.append(d)

            with open(outpath, 'w', encoding='utf-8') as fw:
                json.dump(outdata, fw, indent=2, ensure_ascii=False)
        else:
            fail_count += 1
            print(f'Fail! id: {d["id"]}')

    print(f'success count: {success_count} (exceed count: {exceed_count}), fail count: {fail_count}')


if __name__ == '__main__':
    args = parser.parse_args()
    
    # Debug 模式提示
    if args.debug:
        print("\n" + "="*70)
        print("🐛 DEBUG 模式已启用")
        print(f"   将只测试 {args.debug_samples} 个样本")
        print("="*70 + "\n")
    
    agent_api, model_name = initialize_model(args)

    path = "/mnt/shared-storage-user/xieweiwei/Agent-SafetyBench/score/shield_results/qwen-30B/qwen-30B_gen_res_outputs_unsafe.json"
    
    # 检查文件是否存在
    if not os.path.exists(path):
        print(f"❌ 错误：数据文件不存在: {path}")
        print("请检查路径是否正确")
        sys.exit(1)
    
    outdir = f"./evaluation_results/tot-{model_name}"

    if args.debug:
        outdir += "_debug"  # Debug模式使用单独的输出目录
    
    outdir += f'{args.extra_info}'
    outpath = f"{outdir}/gen_res.json"
    error_log_path = f"{outdir}/error_samples.json"
    exceed_log_path = f"{outdir}/exceed_round_samples.json"
    log_path = f"{outdir}/gen_log.txt"

    os.makedirs(outdir, exist_ok=True)

    print(f"\n配置信息:")
    print(f"  数据路径: {path}")
    print(f"  输出目录: {outdir}")
    print(f"  模型: {model_name}")
    print(f"  Debug模式: {'是' if args.debug else '否'}")
    if args.debug:
        print(f"  测试样本数: {args.debug_samples}")
    print("")

    redirect_output(log_path, 'w')
    regenerate_exceed_samples = args.regen_exceed
    if regenerate_exceed_samples:
        remove_exceed_samples(outpath)
        clear_log(clear_error=True, clear_exceed=True)
    else:
        clear_log(clear_error=True, clear_exceed=False)
    
    # 调用 eval_file，传入 debug 参数
    eval_file(path, outpath, debug_mode=args.debug, debug_samples=args.debug_samples)