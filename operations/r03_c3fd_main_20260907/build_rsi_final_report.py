"""Build the final readable report from completed, hash-checked RSI artifacts."""
import hashlib
import json
from decimal import Decimal,ROUND_HALF_UP
from pathlib import Path
import re

repo=Path(__file__).resolve().parents[2]
p=repo/'docs/r03_paper_story_20260907/execution/post_refine_v08/rsi'
doc=repo/'docs/r03_paper_story_20260907/EXPERT_REPAIR_DISCUSSION.md'
read=lambda name:json.loads((p/name).read_text(encoding='utf-8'))
digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
q=read('RSI_FINAL_RESULTS.json');audit=read('RSI_FINAL_AUDIT.json')
calls=read('RSI_FINAL_CALL_COUNTS.json');cost=read('RSI_FINAL_GPU_ACCOUNTING.json')
manifest=read('RSI_FINAL_ARTIFACTS.json')
assert len(q['stages'])==36 and not q['pending_stages']
assert len(q['training'])==6 and q['complete_joint_weight_updates']==3
assert audit['status']=='passed' and not cost['active_jobs']
assert all(Path(n).name==n and digest(p/n)==h for n,h in manifest['files_sha256'].items())
assert all(x['requested']==256 and all(isinstance(x[k],int) for k in ['comp_valid','Struct_valid','SUN','MSUN']) for x in q['stages'])
stages={(x['cohort'],x['theta'],x['stage']):x for x in q['stages']}
changes={(x['cohort'],x['theta_after'],x['stage']):x for x in q['same_plan_weight_comparisons']}
adjacent=[]
for cohort in ['MAIN','FIT']:
    for stage in ['construction','refined','tokenized','edited']:
        for after in [1,2,3]:
            row=dict(cohort=cohort,stage=stage,theta_before=after-1,theta_after=after,requested=256)
            for metric in ['SUN','MSUN']:
                old=changes.get((cohort,after-1,stage),{})
                new=changes[cohort,after,stage]
                assert old.get(metric+'_unknown_pairs',0)==new[metric+'_unknown_pairs']==0
                og=set(old.get(metric+'_gain_sources',[]));ol=set(old.get(metric+'_loss_sources',[]))
                ng=set(new[metric+'_gain_sources']);nl=set(new[metric+'_loss_sources'])
                gains=(ng-og)|(ol-nl);losses=(og-ng)|(nl-ol)
                assert not gains&losses
                assert len(gains)-len(losses)==stages[cohort,after,stage][metric]-stages[cohort,after-1,stage][metric]
                row.update({metric+'_gains':len(gains),metric+'_losses':len(losses),
                            metric+'_gain_sources':sorted(gains),metric+'_loss_sources':sorted(losses)})
            adjacent.append(row)
paired=dict(source_results_sha256=digest(p/'RSI_FINAL_RESULTS.json'),
            method='Exact set differences of the same-request binary transitions relative to theta0; no unknown pairs.',comparisons=adjacent)
name='RSI_FINAL_PAIRED_CHANGES.json'
(p/name).write_bytes((json.dumps(paired,sort_keys=True,indent=2)+'\n').encode('utf-8'))
manifest['derived_files_sha256']={name:digest(p/name)}
(p/'RSI_FINAL_ARTIFACTS.json').write_bytes((json.dumps(manifest,sort_keys=True,indent=2)+'\n').encode('utf-8'))

labels={'construction':'raw','refined':'浮点 F800','tokenized':'token-F','edited':'KEEP/EDIT 后'}
def percent(count):
    return (Decimal(count)*100/256).quantize(Decimal('0.01'),rounding=ROUND_HALF_UP)

def metric_table(cohort,baselines=False):
    rows=['| 模型／流程 | 阶段 | comp_valid | Struct_valid | SUN | MSUN |',
          '|---|---|---:|---:|---:|---:|']
    selected=[]
    if baselines:
        for arm in ['H1A2','R03']:
            for suffix,label in [('_raw','raw'),('','F800')]:
                selected.append((arm,label,stages['MAIN',0,'baselines/'+arm+suffix]))
    for i in range(4):
        for stage,label in labels.items(): selected.append((f'θ{i}',label,stages[cohort,i,stage]))
    for model,stage,x in selected:
        cells=[f"{x[k]}（{percent(x[k])}%）" for k in ['comp_valid','Struct_valid','SUN','MSUN']]
        rows.append('| '+model+' | '+stage+' | '+' | '.join(cells)+' |')
    return '\n'.join(rows)

pair_rows=['| MAIN 最终输出变化 | SUN 新增／损失 | MSUN 新增／损失 |',
           '|---|---:|---:|']
for x in adjacent:
    if x['cohort']=='MAIN' and x['stage']=='edited':
        pair_rows.append(f"| θ{x['theta_before']} → θ{x['theta_after']} | {x['SUN_gains']}／{x['SUN_losses']} | {x['MSUN_gains']}／{x['MSUN_losses']} |")
train_rows=['| 更新 | 分支 | 实际优化步数 | 参数变化平方和 |','|---|---|---:|---:|']
for x in q['training']:
    train_rows.append(f"| {x['update']} | {x['branch']} | {x['optimizer_steps']} | {x['parameter_delta_squared']:.9f} |")
call_rows=['| 来源 | 版本 | G 前向调用 | 保存的 F800 结果 | SUN 跳过 F | token 回退 raw | E 前向调用 | 实际编辑数 |',
           '|---|---|---:|---:|---:|---:|---:|---:|']
for x in calls['rows']:
    assert x['G_records']==x['F_records']==x['E_records']==256
    call_rows.append(f"| {'TRAIN' if x['cohort']=='FIT' else 'MAIN'} | θ{x['theta']} | {x['G_forwards']} | {x['F800_saved']} | {x['SUN_bypass']} | {x['token_fallback']} | {x['E_forwards']} | {x['E_changed']} |")
link='execution/post_refine_v08/rsi/'
summary=f'''<!-- RSI_FINAL_SUMMARY_BEGIN -->
### 0.9.1 最终结果（2026-09-09）

本轮已完成3次G更新、3次E更新，以及θ0—θ3在同一套256个MAIN Plan和256个TRAIN Plan上的完整四阶段评测。工件中的FIT即TRAIN；训练来源排除了原全量MAIN的1159种约分组成。Plan、种子、K8硬约束及F800设置保持固定，未运行Direct套件。

固定MAIN面板的最终SUN计数为18→21→23→21，MSUN为101→110→114→99。θ2是本面板SUN/MSUN的最佳观察点，θ3出现回退；这次实验未建立稳定自提升的证据。θ3相对θ0最终SUN增加3，MSUN减少2；相对θ2最终SUN减少2，MSUN减少15。三轮真实训练已执行，但不能据此宣称稳定RSI成立。教师model494与物理评分仍是外部反馈来源。

**MAIN：所有分母均为256。** 基线采用固定F800；新流程还包含硬约束生成、raw SUN门控、token回写与编辑。跨基线比较包含流程差异，权重迭代效果应看同流程θ0—θ3的配对。

{metric_table('MAIN',True)}

{chr(10).join(pair_rows)}

第三轮MSUN回落已出现在浮点F阶段（θ2的119降至θ3的100），随后token回写为99。E3在MAIN作出255次KEEP和1次EDIT决策，最终未提交结构修改，保持21个已知SUN，使用269次前向调用。本轮MAIN编辑器未带来额外SUN/MSUN收益。θ3有1条token回写因硬合法性检查回退raw，已计入真实输出与完整分母。

**TRAIN：所有分母均为256，只作为训练反馈，不作为MAIN泛化证据。**

{metric_table('FIT')}

E3在TRAIN实际编辑7条，32个已知SUN的token均保持不变；最终SUN为32、MSUN为116。训练条件上的结果不能替代留出MAIN结果。全部相邻版本、逐阶段新增／损失及原始请求ID见[配对变化]({link}RSI_FINAL_PAIRED_CHANGES.json)。本试验只有一组预固定Plan/seed面板，θ2的最佳观察值也来自该同一面板。

**真实权重更新与计算记录。** G每轮完成128步，E按1200秒优化预算在步边界收尾，每轮实际完成96步；六份检查点均有非零参数变化与完整文件哈希。冻结输入／输出表校验通过。

{chr(10).join(train_rows)}

{chr(10).join(call_rows)}

调用量来自已保存请求记录；F800列是成功保存结果数。TRAIN的E调用包含反事实提案，MAIN列对应实际推理流程。失败或中断但未保存的额外尝试由分配账目单独统计。已登记的本RSI运行共有{len(cost['jobs'])}个作业，含120个完成、4个失败和1个超时分配；最终必要阶段均已得到完整成功结果。六次新训练合计{cost['training_gpu_hours']:.3f} GPU小时，本RSI目录下全部登记作业合计{cost['allocated_gpu_hours']:.3f} GPU小时，历史缓存资产和前序运行准备成本不在该账目中。当前没有遗留的本轮运行作业。

**最终核对与工件。** 已核对36个完整指标结果、9216条阶段记录绑定、2446个F800结果的原种子与800步、2048个SUN门控请求、2048条编辑输出来源及1982个实际编辑条件的健康标记。各版本Plan字节、实际E检查点、token输入权威及冻结IO表均通过检查。后续训练回执的两个描述性轮次字段存在继承首轮值的问题，已保留原回执并给出[更正记录]({link}ROUND_METADATA_ERRATUM.json)；实际父检查点、数据和输出路径均正确。配置生成问题已修复，6项轮次身份测试通过。本轮实验仍使用原冻结源码，模型和评测未因该修复重跑。

[完整36行CSV]({link}RSI_METRICS.csv) · [完整指标与来源哈希]({link}RSI_FINAL_RESULTS.json) · [最终核对]({link}RSI_FINAL_AUDIT.json) · [调用量]({link}RSI_FINAL_CALL_COUNTS.json) · [作业账目]({link}RSI_FINAL_GPU_ACCOUNTING.json) · [工件校验清单]({link}RSI_FINAL_ARTIFACTS.json)
<!-- RSI_FINAL_SUMMARY_END -->
'''
text=doc.read_text(encoding='utf-8')
if '<!-- RSI_FINAL_SUMMARY_BEGIN -->' in text:
    text=re.sub(r'<!-- RSI_FINAL_SUMMARY_BEGIN -->.*?<!-- RSI_FINAL_SUMMARY_END -->',lambda _:summary.rstrip(),text,flags=re.S)
else:
    old=re.search(r'^### 0\.9\.1 初始模型实测与更新结果.*$',text,re.M)
    assert old is not None
    history='### 0.9.2 执行过程记录（保留阶段性观察）\n\n以下保留执行中的阶段快照；最终结果以§0.9.1及最终工件为准。'
    text=text[:old.start()]+summary+'\n'+history+text[old.end():]
text=re.sub(r'^创建日期：.*$',
    '创建日期：2026-09-07。更新日期：2026-09-09。状态：三组G/E真实权重更新及θ0—θ3固定面板评测已全部完成；MAIN的SUN/MSUN在θ2达到本次最高观察值，θ3出现回退。最终结果见§0.9.1，此前讨论与阶段记录保留。',text,count=1,flags=re.M)
doc.write_bytes(text.encode('utf-8'))
print(json.dumps({'metric_rows':len(q['stages']),'adjacent_comparisons':len(adjacent),'final_summary_written':True}))
