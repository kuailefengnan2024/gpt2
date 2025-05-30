import graphviz
from graphviz import Digraph
import os

def create_model_architecture_graph():
    """创建GPT模型架构图"""
    dot = Digraph(comment='GPT模型架构')
    dot.attr(rankdir='TB', size='12,16')
    dot.attr('node', shape='box', style='rounded,filled', fontname='SimHei')
    dot.attr('edge', fontname='SimHei')
    
    # 输入层
    dot.node('input', '输入 tokens\n(B, T)', fillcolor='lightblue')
    
    # Embedding层
    dot.node('wte', 'Token Embedding\n(wte)', fillcolor='lightgreen')
    dot.node('wpe', 'Position Embedding\n(wpe)', fillcolor='lightgreen')
    dot.node('embed_add', 'Token + Position\nEmbedding', fillcolor='lightyellow')
    
    # Transformer Blocks
    dot.node('blocks', 'Transformer Blocks\n(n_layer=12)', fillcolor='orange')
    
    # Block内部结构
    with dot.subgraph(name='cluster_block') as block:
        block.attr(label='单个Transformer Block', fontname='SimHei')
        block.node('ln1', 'LayerNorm 1', fillcolor='lightcoral')
        block.node('attention', 'Multi-Head\nSelf-Attention', fillcolor='lightcoral')
        block.node('add1', 'Residual Add 1', fillcolor='lightyellow')
        block.node('ln2', 'LayerNorm 2', fillcolor='lightcoral')
        block.node('mlp', 'MLP\n(4*n_embd)', fillcolor='lightcoral')
        block.node('add2', 'Residual Add 2', fillcolor='lightyellow')
    
    # Attention内部结构
    with dot.subgraph(name='cluster_attention') as attn:
        attn.attr(label='Multi-Head Self-Attention', fontname='SimHei')
        attn.node('qkv', 'Linear\n(3*n_embd)', fillcolor='pink')
        attn.node('split', 'Split Q,K,V', fillcolor='pink')
        attn.node('reshape', 'Reshape\n(B,nh,T,hs)', fillcolor='pink')
        attn.node('flash_attn', 'Flash Attention\n(scaled_dot_product)', fillcolor='pink')
        attn.node('concat', 'Concat Heads', fillcolor='pink')
        attn.node('proj', 'Output Projection', fillcolor='pink')
    
    # MLP内部结构
    with dot.subgraph(name='cluster_mlp') as mlp_graph:
        mlp_graph.attr(label='MLP', fontname='SimHei')
        mlp_graph.node('fc', 'Linear\n(4*n_embd)', fillcolor='lightsteelblue')
        mlp_graph.node('gelu', 'GELU激活', fillcolor='lightsteelblue')
        mlp_graph.node('proj_mlp', 'Linear\n(n_embd)', fillcolor='lightsteelblue')
    
    # 输出层
    dot.node('ln_f', 'Final LayerNorm', fillcolor='lightgreen')
    dot.node('lm_head', 'Language Model Head\n(Linear to vocab_size)', fillcolor='lightgreen')
    dot.node('output', '输出 Logits\n(B, T, vocab_size)', fillcolor='lightblue')
    
    # 连接主要流程
    dot.edge('input', 'wte')
    dot.edge('input', 'wpe')
    dot.edge('wte', 'embed_add')
    dot.edge('wpe', 'embed_add')
    dot.edge('embed_add', 'blocks')
    dot.edge('blocks', 'ln_f')
    dot.edge('ln_f', 'lm_head')
    dot.edge('lm_head', 'output')
    
    return dot

def create_training_flow_graph():
    """创建训练流程图"""
    dot = Digraph(comment='GPT训练流程')
    dot.attr(rankdir='TB', size='14,20')
    dot.attr('node', shape='box', style='rounded,filled', fontname='SimHei')
    dot.attr('edge', fontname='SimHei')
    
    # 初始化阶段
    dot.node('start', '开始训练', fillcolor='lightgreen')
    dot.node('init_ddp', '初始化DDP\n(如果多GPU)', fillcolor='lightblue')
    dot.node('init_model', '初始化GPT模型', fillcolor='lightblue')
    dot.node('init_data', '初始化数据加载器\n(train/val)', fillcolor='lightblue')
    dot.node('init_opt', '初始化优化器\n(AdamW)', fillcolor='lightblue')
    
    # 主训练循环
    dot.node('main_loop', '主训练循环\n(max_steps=19073)', fillcolor='orange', shape='ellipse')
    
    # 验证评估
    dot.node('check_val', '是否进行验证？\n(每250步)', fillcolor='yellow', shape='diamond')
    dot.node('val_eval', '验证评估', fillcolor='lightcoral')
    dot.node('val_loss', '计算验证损失', fillcolor='lightcoral')
    dot.node('save_ckpt', '保存checkpoint\n(每5000步)', fillcolor='lightcoral')
    
    # HellaSwag评估
    dot.node('check_hella', '是否进行HellaSwag？\n(每250步)', fillcolor='yellow', shape='diamond')
    dot.node('hella_eval', 'HellaSwag评估', fillcolor='lightpink')
    dot.node('hella_acc', '计算准确率', fillcolor='lightpink')
    
    # 文本生成
    dot.node('check_gen', '是否生成文本？\n(每250步)', fillcolor='yellow', shape='diamond')
    dot.node('text_gen', '文本生成\n(top-k采样)', fillcolor='lightsteelblue')
    
    # 训练步骤
    dot.node('train_step', '训练步骤', fillcolor='lightyellow')
    dot.node('zero_grad', '梯度清零', fillcolor='lightyellow')
    dot.node('grad_accum', '梯度累积循环\n(micro batches)', fillcolor='lightyellow')
    dot.node('forward', '前向传播', fillcolor='lightyellow')
    dot.node('loss_calc', '损失计算\n(交叉熵)', fillcolor='lightyellow')
    dot.node('backward', '反向传播', fillcolor='lightyellow')
    dot.node('grad_clip', '梯度裁剪', fillcolor='lightyellow')
    dot.node('lr_schedule', '学习率调度\n(余弦衰减)', fillcolor='lightyellow')
    dot.node('optimizer_step', '优化器步骤', fillcolor='lightyellow')
    
    # 数据加载
    dot.node('load_batch', '加载batch\n(DataLoaderLite)', fillcolor='lightgreen')
    dot.node('to_device', '数据移到设备\n(GPU/CPU)', fillcolor='lightgreen')
    
    # 结束
    dot.node('check_end', '训练完成？', fillcolor='yellow', shape='diamond')
    dot.node('cleanup', '清理资源\n(DDP销毁)', fillcolor='lightgreen')
    dot.node('end', '训练结束', fillcolor='lightgreen')
    
    # 连接流程
    dot.edge('start', 'init_ddp')
    dot.edge('init_ddp', 'init_model')
    dot.edge('init_model', 'init_data')
    dot.edge('init_data', 'init_opt')
    dot.edge('init_opt', 'main_loop')
    
    # 主循环内部
    dot.edge('main_loop', 'check_val')
    dot.edge('check_val', 'val_eval', label='是')
    dot.edge('check_val', 'check_hella', label='否')
    dot.edge('val_eval', 'val_loss')
    dot.edge('val_loss', 'save_ckpt')
    dot.edge('save_ckpt', 'check_hella')
    
    dot.edge('check_hella', 'hella_eval', label='是')
    dot.edge('check_hella', 'check_gen', label='否')
    dot.edge('hella_eval', 'hella_acc')
    dot.edge('hella_acc', 'check_gen')
    
    dot.edge('check_gen', 'text_gen', label='是')
    dot.edge('check_gen', 'train_step', label='否')
    dot.edge('text_gen', 'train_step')
    
    # 训练步骤详细流程
    dot.edge('train_step', 'zero_grad')
    dot.edge('zero_grad', 'grad_accum')
    dot.edge('grad_accum', 'load_batch')
    dot.edge('load_batch', 'to_device')
    dot.edge('to_device', 'forward')
    dot.edge('forward', 'loss_calc')
    dot.edge('loss_calc', 'backward')
    dot.edge('backward', 'grad_accum', label='下一个micro batch')
    dot.edge('grad_accum', 'grad_clip', label='累积完成')
    dot.edge('grad_clip', 'lr_schedule')
    dot.edge('lr_schedule', 'optimizer_step')
    dot.edge('optimizer_step', 'check_end')
    
    dot.edge('check_end', 'main_loop', label='否')
    dot.edge('check_end', 'cleanup', label='是')
    dot.edge('cleanup', 'end')
    
    return dot

def create_data_flow_graph():
    """创建数据流图"""
    dot = Digraph(comment='数据流图')
    dot.attr(rankdir='LR', size='16,10')
    dot.attr('node', shape='box', style='rounded,filled', fontname='SimHei')
    dot.attr('edge', fontname='SimHei')
    
    # 数据源
    dot.node('data_source', '数据源\nedu_fineweb10B', fillcolor='lightblue')
    dot.node('shards', 'Shard文件\n(.bin格式)', fillcolor='lightblue')
    
    # DataLoader
    dot.node('dataloader', 'DataLoaderLite', fillcolor='lightgreen')
    dot.node('load_tokens', 'load_tokens()\nnumpy -> torch', fillcolor='lightgreen')
    dot.node('batch_slice', 'Batch切片\n(B, T)', fillcolor='lightgreen')
    
    # 数据处理
    dot.node('inputs', '输入序列 x\n(B, T)', fillcolor='lightyellow')
    dot.node('targets', '目标序列 y\n(B, T)', fillcolor='lightyellow')
    dot.node('device_move', '移动到设备\n(GPU/CPU)', fillcolor='lightyellow')
    
    # 模型处理
    dot.node('model_forward', 'GPT前向传播', fillcolor='orange')
    dot.node('logits_out', 'Logits输出\n(B, T, vocab_size)', fillcolor='orange')
    dot.node('loss_fn', '损失函数\n(交叉熵)', fillcolor='orange')
    dot.node('loss_value', '损失值', fillcolor='orange')
    
    # Token化处理
    with dot.subgraph(name='cluster_tokenization') as tok:
        tok.attr(label='Token化处理', fontname='SimHei')
        tok.node('tiktoken', 'tiktoken编码器\n(GPT-2 BPE)', fillcolor='lightcoral')
        tok.node('vocab', '词汇表\n(50304 tokens)', fillcolor='lightcoral')
    
    # 连接数据流
    dot.edge('data_source', 'shards')
    dot.edge('shards', 'dataloader')
    dot.edge('dataloader', 'load_tokens')
    dot.edge('load_tokens', 'batch_slice')
    dot.edge('batch_slice', 'inputs')
    dot.edge('batch_slice', 'targets', label='偏移1位')
    dot.edge('inputs', 'device_move')
    dot.edge('targets', 'device_move')
    dot.edge('device_move', 'model_forward')
    dot.edge('model_forward', 'logits_out')
    dot.edge('logits_out', 'loss_fn')
    dot.edge('targets', 'loss_fn')
    dot.edge('loss_fn', 'loss_value')
    
    return dot

def create_evaluation_flow_graph():
    """创建评估流程图"""
    dot = Digraph(comment='评估流程')
    dot.attr(rankdir='TB', size='12,14')
    dot.attr('node', shape='box', style='rounded,filled', fontname='SimHei')
    dot.attr('edge', fontname='SimHei')
    
    # 验证损失评估
    with dot.subgraph(name='cluster_val') as val:
        val.attr(label='验证损失评估', fontname='SimHei')
        val.node('val_start', '设置模型为评估模式', fillcolor='lightblue')
        val.node('val_reset', '重置验证数据加载器', fillcolor='lightblue')
        val.node('val_loop', '验证循环\n(20个batch)', fillcolor='lightblue')
        val.node('val_forward', '前向传播\n(无梯度)', fillcolor='lightblue')
        val.node('val_loss_acc', '累积验证损失', fillcolor='lightblue')
        val.node('val_ddp_reduce', 'DDP损失聚合', fillcolor='lightblue')
        val.node('val_log', '记录验证损失', fillcolor='lightblue')
    
    # HellaSwag评估
    with dot.subgraph(name='cluster_hella') as hella:
        hella.attr(label='HellaSwag评估', fontname='SimHei')
        hella.node('hella_start', '开始HellaSwag', fillcolor='lightgreen')
        hella.node('hella_examples', '遍历验证样本', fillcolor='lightgreen')
        hella.node('hella_render', '渲染样本\n(tokens, mask, label)', fillcolor='lightgreen')
        hella.node('hella_forward', '模型前向传播', fillcolor='lightgreen')
        hella.node('hella_most_likely', '找到最可能的补全\n(最低损失)', fillcolor='lightgreen')
        hella.node('hella_count', '统计正确数量', fillcolor='lightgreen')
        hella.node('hella_ddp_reduce', 'DDP统计聚合', fillcolor='lightgreen')
        hella.node('hella_acc', '计算准确率', fillcolor='lightgreen')
        hella.node('hella_log', '记录准确率', fillcolor='lightgreen')
    
    # 文本生成
    with dot.subgraph(name='cluster_gen') as gen:
        gen.attr(label='文本生成', fontname='SimHei')
        gen.node('gen_start', '设置生成参数\n(max_length=32)', fillcolor='lightyellow')
        gen.node('gen_prompt', '初始提示\n"Hello, I\'m a language model,"', fillcolor='lightyellow')
        gen.node('gen_loop', '生成循环', fillcolor='lightyellow')
        gen.node('gen_forward', '模型前向传播', fillcolor='lightyellow')
        gen.node('gen_logits', '获取最后位置logits', fillcolor='lightyellow')
        gen.node('gen_softmax', '计算概率分布', fillcolor='lightyellow')
        gen.node('gen_topk', 'Top-K采样\n(k=50)', fillcolor='lightyellow')
        gen.node('gen_sample', '多项式采样', fillcolor='lightyellow')
        gen.node('gen_append', '追加到序列', fillcolor='lightyellow')
        gen.node('gen_decode', '解码并打印', fillcolor='lightyellow')
    
    # 连接评估流程
    # 验证损失流程
    dot.edge('val_start', 'val_reset')
    dot.edge('val_reset', 'val_loop')
    dot.edge('val_loop', 'val_forward')
    dot.edge('val_forward', 'val_loss_acc')
    dot.edge('val_loss_acc', 'val_loop', label='下一个batch')
    dot.edge('val_loss_acc', 'val_ddp_reduce', label='完成')
    dot.edge('val_ddp_reduce', 'val_log')
    
    # HellaSwag流程
    dot.edge('hella_start', 'hella_examples')
    dot.edge('hella_examples', 'hella_render')
    dot.edge('hella_render', 'hella_forward')
    dot.edge('hella_forward', 'hella_most_likely')
    dot.edge('hella_most_likely', 'hella_count')
    dot.edge('hella_count', 'hella_examples', label='下一个样本')
    dot.edge('hella_count', 'hella_ddp_reduce', label='完成')
    dot.edge('hella_ddp_reduce', 'hella_acc')
    dot.edge('hella_acc', 'hella_log')
    
    # 文本生成流程
    dot.edge('gen_start', 'gen_prompt')
    dot.edge('gen_prompt', 'gen_loop')
    dot.edge('gen_loop', 'gen_forward')
    dot.edge('gen_forward', 'gen_logits')
    dot.edge('gen_logits', 'gen_softmax')
    dot.edge('gen_softmax', 'gen_topk')
    dot.edge('gen_topk', 'gen_sample')
    dot.edge('gen_sample', 'gen_append')
    dot.edge('gen_append', 'gen_loop', label='继续生成')
    dot.edge('gen_append', 'gen_decode', label='达到最大长度')
    
    return dot

def main():
    """主函数：生成所有流程图"""
    # 创建输出目录
    os.makedirs('flowcharts', exist_ok=True)
    
    # print("生成GPT模型架构图...")
    # model_graph = create_model_architecture_graph()
    # model_graph.render('flowcharts/gpt_model_architecture', format='pdf', cleanup=True)
    
    print("生成训练流程图...")
    training_graph = create_training_flow_graph()
    training_graph.render('flowcharts/gpt_training_flow', format='pdf', cleanup=True)
    
    print("生成数据流图...")
    data_graph = create_data_flow_graph()
    data_graph.render('flowcharts/gpt_data_flow', format='pdf', cleanup=True)
    
    print("生成评估流程图...")
    eval_graph = create_evaluation_flow_graph()
    eval_graph.render('flowcharts/gpt_evaluation_flow', format='pdf', cleanup=True)
    
    # 创建综合流程图
    print("生成综合流程图...")
    comprehensive_graph = create_comprehensive_flow_graph()
    comprehensive_graph.render('flowcharts/gpt_comprehensive_flow', format='pdf', cleanup=True)
    
    print("所有流程图已生成完成！")
    print("输出文件位置：flowcharts/ 目录")
    print("生成的文件：")
    # print("  - gpt_model_architecture.pdf - GPT模型架构图")
    print("  - gpt_training_flow.pdf - 训练流程图")
    print("  - gpt_data_flow.pdf - 数据流图")
    print("  - gpt_evaluation_flow.pdf - 评估流程图")
    print("  - gpt_comprehensive_flow.pdf - 综合流程图")

def create_comprehensive_flow_graph():
    """创建综合流程图"""
    dot = Digraph(comment='GPT训练综合流程')
    dot.attr(rankdir='TB', size='20,24')
    dot.attr('node', shape='box', style='rounded,filled', fontname='SimHei')
    dot.attr('edge', fontname='SimHei')
    
    # 设置不同阶段的颜色
    init_color = 'lightblue'
    data_color = 'lightgreen'
    model_color = 'orange'
    train_color = 'lightyellow'
    eval_color = 'lightcoral'
    save_color = 'lightpink'
    
    # 初始化阶段
    with dot.subgraph(name='cluster_init') as init:
        init.attr(label='初始化阶段', fontname='SimHei', style='filled', color='lightgray')
        init.node('start', '开始', fillcolor=init_color)
        init.node('setup_ddp', '设置分布式训练\n(DDP)', fillcolor=init_color)
        init.node('create_model', '创建GPT模型\nvocab_size=50304', fillcolor=init_color)
        init.node('setup_data', '设置数据加载器\nedu_fineweb10B', fillcolor=init_color)
        init.node('setup_optimizer', '设置优化器\nAdamW', fillcolor=init_color)
        init.node('setup_lr', '设置学习率调度\n余弦衰减+warmup', fillcolor=init_color)
    
    # 主训练循环
    dot.node('main_loop', '主训练循环\n(19073 steps)', fillcolor=model_color, shape='ellipse')
    
    # 数据处理
    with dot.subgraph(name='cluster_data') as data:
        data.attr(label='数据处理', fontname='SimHei', style='filled', color='lightgray')
        data.node('load_batch', '加载batch\n(B=64, T=1024)', fillcolor=data_color)
        data.node('tokenize', 'Token化处理\ntiktoken GPT-2', fillcolor=data_color)
        data.node('to_device', '移动到设备', fillcolor=data_color)
    
    # 模型前向传播
    with dot.subgraph(name='cluster_forward') as forward:
        forward.attr(label='模型前向传播', fontname='SimHei', style='filled', color='lightgray')
        forward.node('embedding', 'Token + Position\nEmbedding', fillcolor=model_color)
        forward.node('transformer', 'Transformer Blocks\n(12层)', fillcolor=model_color)
        forward.node('attention', 'Multi-Head\nSelf-Attention', fillcolor=model_color)
        forward.node('mlp', 'MLP\n(4*n_embd)', fillcolor=model_color)
        forward.node('output_head', 'Language Model Head', fillcolor=model_color)
        forward.node('logits', 'Logits输出\n(B,T,vocab_size)', fillcolor=model_color)
    
    # 训练步骤
    with dot.subgraph(name='cluster_training') as training:
        training.attr(label='训练步骤', fontname='SimHei', style='filled', color='lightgray')
        training.node('calc_loss', '计算损失\n交叉熵', fillcolor=train_color)
        training.node('backward', '反向传播', fillcolor=train_color)
        training.node('grad_accum', '梯度累积\n(grad_accum_steps)', fillcolor=train_color)
        training.node('grad_clip', '梯度裁剪\n(norm=1.0)', fillcolor=train_color)
        training.node('update_lr', '更新学习率', fillcolor=train_color)
        training.node('optimizer_step', '优化器更新', fillcolor=train_color)
    
    # 评估阶段
    with dot.subgraph(name='cluster_eval') as eval_cluster:
        eval_cluster.attr(label='评估阶段 (每250步)', fontname='SimHei', style='filled', color='lightgray')
        eval_cluster.node('val_loss', '验证损失评估', fillcolor=eval_color)
        eval_cluster.node('hellaswag', 'HellaSwag评估', fillcolor=eval_color)
        eval_cluster.node('text_gen', '文本生成\nTop-K采样', fillcolor=eval_color)
    
    # 保存和日志
    with dot.subgraph(name='cluster_save') as save:
        save.attr(label='保存和日志', fontname='SimHei', style='filled', color='lightgray')
        save.node('log_metrics', '记录指标\nloss, lr, accuracy', fillcolor=save_color)
        save.node('save_checkpoint', '保存checkpoint\n(每5000步)', fillcolor=save_color)
    
    # 决策节点
    dot.node('check_step', '检查步数', fillcolor='yellow', shape='diamond')
    dot.node('check_eval', '是否评估？\n(step % 250 == 0)', fillcolor='yellow', shape='diamond')
    dot.node('check_save', '是否保存？\n(step % 5000 == 0)', fillcolor='yellow', shape='diamond')
    dot.node('check_end', '训练完成？\n(step >= max_steps)', fillcolor='yellow', shape='diamond')
    
    # 结束
    dot.node('cleanup', '清理资源', fillcolor=init_color)
    dot.node('end', '训练结束', fillcolor=init_color)
    
    # 连接主要流程
    # 初始化流程
    dot.edge('start', 'setup_ddp')
    dot.edge('setup_ddp', 'create_model')
    dot.edge('create_model', 'setup_data')
    dot.edge('setup_data', 'setup_optimizer')
    dot.edge('setup_optimizer', 'setup_lr')
    dot.edge('setup_lr', 'main_loop')
    
    # 主循环
    dot.edge('main_loop', 'check_step')
    dot.edge('check_step', 'load_batch')
    
    # 数据处理流程
    dot.edge('load_batch', 'tokenize')
    dot.edge('tokenize', 'to_device')
    
    # 前向传播流程
    dot.edge('to_device', 'embedding')
    dot.edge('embedding', 'transformer')
    dot.edge('transformer', 'attention')
    dot.edge('attention', 'mlp')
    dot.edge('mlp', 'output_head')
    dot.edge('output_head', 'logits')
    
    # 训练流程
    dot.edge('logits', 'calc_loss')
    dot.edge('calc_loss', 'backward')
    dot.edge('backward', 'grad_accum')
    dot.edge('grad_accum', 'grad_clip')
    dot.edge('grad_clip', 'update_lr')
    dot.edge('update_lr', 'optimizer_step')
    
    # 评估检查
    dot.edge('optimizer_step', 'check_eval')
    dot.edge('check_eval', 'val_loss', label='是')
    dot.edge('val_loss', 'hellaswag')
    dot.edge('hellaswag', 'text_gen')
    dot.edge('text_gen', 'log_metrics')
    dot.edge('check_eval', 'check_save', label='否')
    
    # 保存检查
    dot.edge('log_metrics', 'check_save')
    dot.edge('check_save', 'save_checkpoint', label='是')
    dot.edge('save_checkpoint', 'check_end')
    dot.edge('check_save', 'check_end', label='否')
    
    # 循环控制
    dot.edge('check_end', 'main_loop', label='否\n继续训练')
    dot.edge('check_end', 'cleanup', label='是\n完成')
    dot.edge('cleanup', 'end')
    
    return dot

if __name__ == "__main__":
    main() 