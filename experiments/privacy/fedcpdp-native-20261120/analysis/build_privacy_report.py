import json
from pathlib import Path
from docx import Document
from docx.shared import Inches,Pt,RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
O=Path(__file__).resolve().parents[4]/'results/fedcpdp-privacy-utility-20261120/report';a=json.loads((O/'analysis.json').read_text(encoding='utf-8'))
cs=a['conditions'];names=['无扰动','仅裁剪','σ＝0.1','σ＝0.25','σ＝0.5','σ＝1.0'];labels=dict(zip(cs,names));m=a['manifest']
d=Document();sec=d.sections[0];sec.page_width=Inches(8.5);sec.page_height=Inches(11)
sec.top_margin=sec.bottom_margin=Inches(.65);sec.left_margin=sec.right_margin=Inches(.65)
for n in ['Normal','Title','Heading 1','Heading 2']:
    st=d.styles[n];st.font.name='Calibri';st._element.rPr.rFonts.set(qn('w:eastAsia'),'宋体');st.font.color.rgb=RGBColor(0,0,0)
    st.font.size=Pt(11 if n=='Normal' else 21 if n=='Title' else 15 if n=='Heading 1' else 12)
    st.paragraph_format.space_after=Pt(6);st.paragraph_format.line_spacing=1.12
sec.header.paragraphs[0].text='FedCPDP 同构场景隐私扰动实验报告'
sec.header.paragraphs[0].style='Caption';sec.header.paragraphs[0].runs[0].font.color.rgb=RGBColor(0,0,0)
p=sec.footer.paragraphs[0];p.alignment=2;p.add_run('第 ');fld=OxmlElement('w:fldSimple');fld.set(qn('w:instr'),'PAGE');p._p.append(fld);p.add_run(' 页')
def para(t):return d.add_paragraph(t)
def h(t):d.add_heading(t,1)
def page(t):d.add_page_break();h(t)
def table(headers,rows,size=10.5):
    t=d.add_table(rows=1, cols=len(headers));t.autofit=False
    for c,s in zip(t.rows[0].cells,headers):c.text=s
    for row in rows:
        for c,s in zip(t.add_row().cells,row):c.text=str(s)
    for i,row in enumerate(t.rows):
        for cell in row.cells:
            tcPr=cell._tc.get_or_add_tcPr();mar=OxmlElement('w:tcMar')
            for side in ['top','left','bottom','right']:
                v=OxmlElement('w:'+side);v.set(qn('w:w'),'35' if side in ['top','bottom'] else '65');v.set(qn('w:type'),'dxa');mar.append(v)
            tcPr.append(mar)
            if i==0:
                sh=OxmlElement('w:shd');sh.set(qn('w:fill'),'EAEAEA');tcPr.append(sh)
            for p in cell.paragraphs:
                p.paragraph_format.space_after=Pt(1);p.paragraph_format.space_before=Pt(0);p.paragraph_format.line_spacing=1.0
                for r in p.runs:r.font.size=Pt(size);r.bold=i==0
        pr=row._tr.get_or_add_trPr();pr.append(OxmlElement('w:cantSplit'))
    t.rows[0]._tr.get_or_add_trPr().append(OxmlElement('w:tblHeader'))
    gap=d.add_paragraph();gap.paragraph_format.space_after=Pt(4);gap.paragraph_format.line_spacing=Pt(1);gap.add_run().font.size=Pt(1)
    return t
def f(x):return f'{x:.4f}'
def mean(c,k):return a['overall'][c][k]['mean']

d.add_paragraph('FedCPDP 隐私扰动与预测效用实验报告','Title')
para('同构跨项目缺陷预测　五个数据家族　六种扰动条件')
para('报告日期  2026年11月23日　　实验批次  fedcpdp-privacy-native-20261120')
h('一 实验目的与主要结论')
para('本实验考察在保留 FedCPDP 作者实现的模型、局部训练、对齐和混合聚合机制时，关闭扰动、仅裁剪以及逐步提高高斯噪声乘子，会如何影响同构跨项目缺陷预测的效用。实验使用五个家族的30个目标项目，每个条件运行10个随机种子，共完成1800次训练；全部达到预定50轮，任务失败数为0。')
para('在本次预定网格中，σ＝0.1 的项目宏平均 AUC 为0.6896、MCC为0.2253，均高于无扰动基线的0.6652和0.2145；但经过多重比较校正后，这两项提升不显著。σ提高到0.25及以上时，AUC、MCC和G-mean明显下降。σ＝1.0时，平均AUC为0.5144，已接近随机排序参考值0.5。')
table(['条件','AUC','MCC','Recall','G-mean','F1'],[[labels[c]]+[f(mean(c,k)) for k in ['ROC_AUC','MCC','Recall','G_mean','F1']] for c in cs])
para('表1　先在每个目标项目内对10个种子取平均，再对30个项目等权平均。该表不是将全部测试样本合并计算，也不是五个家族等权平均。')
para('本报告的结论范围是原生扰动机制的效用影响。噪声乘子σ不是隐私预算ε；本次没有完成端到端隐私核算，也未实施成员推断或梯度重建攻击。因此不能据此给出相同ε下的跨方法排名，或声称已经验证完整隐私保护。')
h('二 完成与核验情况')
para('服务器训练与结果复核于2026年11月23日05时28分完成。每种条件均有300份结果，包含预测、验证集预测、阈值、分类指标、工作量指标、日志及检查点。下载归档SHA-256一致，431个冻结文件校验通过；全部1800份结果结构完整。零任务失败不意味着训练没有数值饱和，相关诊断见第七部分。')

page('三 数据集与划分协议')
families=['AEEEM','NASA','PROMISE','Relink','SOFTLAB']
table(['家族','项目数','样本数','项目名单'],[[fam,sum(v['family']==fam for v in m.values()),sum(v['n'] for v in m.values() if v['family']==fam),'、'.join(t for t in sorted(m) if m[t]['family']==fam)] for fam in families])
para(f'冻结数据共{a["data_total"]}条记录。数据来自 FedCPDP 作者发布的数据包，固定版本为 ab5af637a60aed4c34a5338709b3ac65b344ec37；每个CSV的SHA-256、标签数量、特征模式和LOC有效性均保存在 data_manifest.json。这里保留作者发布数据的实际数量，不为贴合论文表格而改动样本。')
para('采用家族内留一项目测试：每次将一个项目作为目标，其余同家族项目各对应一个源客户端。不同家族之间不混合训练。AEEEM、NASA和SOFTLAB任务各有4个源客户端，PROMISE有11个，Relink有2个。每个项目采用冻结的一份CSV；本实验不重新选版本。')
para('源项目沿用冻结v3行号划分，约80%用于训练、约20%用于验证。相同特征记录在客户端内部成组划分，比例因分组可能偏离精确80/20。保留原始文件，在每个任务中排除与目标特征完全一致的源记录；没有实施跨源客户端联合重复分组，所以不同客户端之间仍可能出现训练与验证特征重复。')
para('目标项目全部用于测试，不使用目标标签训练、选择模型或选择阈值。源端标准化仅在该客户端训练部分拟合；目标项目使用自身无标签特征计算标准化统计量，因此包含目标无标签特征访问，属于传导式预处理条件。六种扰动条件共享完全相同的数据和行号划分。')
para('保留作者的数值特征选择和特征哈希。原生加载器按规则排除识别到的工作量列，未统一排除所有可能代表LOC的替代列；具体特征名单逐任务保存。不得将本实验描述为与其他方法使用了完全一致的特征预处理。')
para('分类指标覆盖30个项目。工作量指标只覆盖LOC全部有效的13个项目；其余17个项目记为N/A，不记为0。有效项目包括：'+ '、'.join(t for t in sorted(m) if m[t]['invalid_loc']==0)+'。')

page('四 模型训练与执行配置')
table(['配置项','实际设置'],[
['模型与输入','特征哈希1024维；两层MLP，隐藏层128和64；二分类输出'],['通信与参与','50次实际联邦更新；每轮全部源客户端参与'],['本地训练','每轮5个epoch；batch size 32；不使用跨种子集成'],['随机重复','训练种子0至9；固定数据划分；PYTHONHASHSEED＝0'],['优化器','Adam；初始学习率0.003；β1＝0.9，β2＝0.999，ε＝1e-8'],['学习率调度','每轮本地训练内部按训练步进行余弦调度'],['正则与损失','Dropout 0.2；L2 1e-4；Focal loss γ＝2'],['近端项与对齐','FedProx μ＝0.01；均值和方差对齐权重均为0.001'],['混合聚合','hybrid；数据规模、训练损失与更新方向、表示统计相似度共同作用'],['聚合细节','注意力α＝2、β＝0.5；动量EMA＝0.9；图均值系数1、方差系数0.5'],['模型选择','只采用预定最后一轮；不按目标测试指标选最佳轮次'],['主阈值','0.05至0.95共37点；最大化源客户端宏平均MCC'],['阈值并列规则','优先更高源验证Recall，再选较低阈值'],['补充阈值','另存固定0.5及源验证合并F1最优阈值结果'],['服务器','阿里云ecs.g8i.xlarge；4 vCPU、16 GiB；Ubuntu 24.04 x86_64'],['实际执行','Python 3.12环境；2个并行任务；每任务BLAS单线程'],['依赖','NumPy 2.3.3；pandas 2.3.3；SciPy 1.16.2；scikit-learn 1.7.2']])
para('配置来源需要区分：冻结协议标注通信50轮、本地5 epoch、batch 32、近端项及对齐权重已与论文核对；模型宽度、哈希维度、学习率、Dropout、C和σ的起点继承代码设置。六档扫描与统一验证协议属于本研究设计，不冒充原论文完整实验网格。')
para('protocol.json中的environment字段继承了早期Windows运行记录，并非本次云端运行环境。本报告采用实际云端执行记录和environment.freeze.txt说明环境。这一元数据差异保留披露，未修改冻结训练配置。')

page('五 隐私扰动条件与控制变量')
table(['条件','USE_DP','裁剪C','噪声乘子σ','作用'],[['无扰动','false','不启用','不启用','原始客户端更新'],['仅裁剪','true','1.0','0','只限制更新范数']]+[[labels[c],'true','1.0',str(s),'裁剪后加入高斯噪声'] for c,s in zip(cs[2:],[.1,.25,.5,1.])])
para('每个客户端先完成本地训练，再计算相对全局模型的参数增量。开启扰动时，将全部参数增量展平成一个向量，按全局L2范数进行裁剪，使其范数不超过C；随后向每个坐标加入均值0、标准差σ乘以C的独立高斯噪声。将处理后的更新送入原有混合聚合器。')
para('无扰动条件绕过裁剪与加噪；仅裁剪条件走同一原生处理函数，但噪声标准差为0。前者用于估计完整扰动带来的变化，后者用于区分裁剪本身的影响与裁剪后加噪的影响。除这几个开关和参数外，其余算法、预算、数据和评价规则固定。')
para('作者核心源码没有被替换为新的聚合算法；训练损失仍来自local_train，未替换成验证损失。上传并参与聚合的训练损失、隐藏表示均值和方差并未被完整保护，且聚合权重依赖数据相关信息。仅对模型更新加噪不能据此推出整个算法的端到端差分隐私保证。')
para('不报告ε或δ，也不将σ的数值与其他方法的ε直接比较。本研究未建立保护单位、相邻数据集定义、全部发布通道和多轮组合核算的完整证明。随着σ增大，模型更新的扰动增强是实现层面的事实，不能等同于已经测得真实隐私泄漏降低。')
para('执行顺序为：先在ar1、seed＝0上完成六种条件各50轮的试运行并校验；再扩展到全部目标和种子。云端同包内的试运行计入正式矩阵，未直接复用旧Windows实验结果。服务器停机后从逐轮检查点恢复，已完成任务按配置指纹跳过；恢复不改变科学配置。')

page('六 总体效用与家族差异')
d.add_picture(str(O/'utility.png'),width=Inches(7.05))
para('图1　30个项目的宏平均分类性能。横轴前两项为对照条件，后四项为固定C＝1下的噪声乘子；连线仅帮助阅读，不代表连续隐私预算曲线。')
table(['条件','CostEffect@20','Popt@20','有效项目'],[[labels[c],f(mean(c,'CostEffect@20')),f(mean(c,'Popt@20')),13] for c in cs])
para('工作量评价按预测概率除以原始LOC排序，预算为总LOC的20%，使用线性插值；CostEffect表示该预算内捕获的缺陷模块比例，Popt比较预算区间内的面积与最优、最差排序面积。它们不是运行时间或计算效率指标。完整产出还保存5%、10%和40%预算。')
para('σ＝0.1在本次网格中获得最高总体平均分，AUC相对无扰动增加0.0244，MCC增加0.0108；相对仅裁剪则分别增加0.0302和0.0188。轻度噪声可能具有正则化效果，但本实验未独立检验这一机制，不能将其作为已证实原因。')
para('σ＝0.25起，多项指标明显恶化。Recall并非单调下降，σ＝0.5时Recall仍为0.5109，但MCC只有0.0553，说明不能把召回率单独作为效用保留的证据。高噪声下模型的区分能力和类别平衡表现同时受损。')

page('七 家族层面结果与数值诊断')
for metric,title in [('ROC_AUC','表3　各家族项目宏平均 AUC'),('MCC','表4　各家族项目宏平均 MCC')]:
    para(title);table(['家族']+names,[[fam]+[f(a['family'][c][fam][metric]) for c in cs] for fam in families],10)
para('σ＝0.1时，AEEEM、NASA和PROMISE的家族平均AUC提高，而Relink从0.6671降至0.6075，SOFTLAB从0.7637降至0.6653。因此总体平均提升不代表各家族均受益。家族内部按项目等权平均；其样本量、类别比例、特征和客户端数量不同，结果应结合附录逐项目检查。')
table(['条件','含exp溢出警告的运行数','极端概率比例','边界阈值运行数'],[[labels[c],a['diagnostics'][c]['overflow_warning_runs'],f"{100*a['diagnostics'][c]['mean_extreme_probability_fraction']:.2f}%",a['diagnostics'][c]['boundary_threshold_runs']] for c in cs],10)
para('表5　每种条件分母均为300次运行。极端概率指p≤10⁻⁶或p≥1−10⁻⁶，先计算每次运行的测试样本比例再平均；边界阈值指选中0.05或0.95。警告数按每个运行日志是否出现overflow encountered in exp统计，不是警告行数。')
para('σ＝0.5和1.0的全部运行均出现exp溢出警告，极端概率比例分别约99.72%和99.98%。服务器仍完成有限参数与预测检查，但数值饱和是重要的质量限制。高噪声性能下降同时伴随原始数值实现的饱和，不能把全部下降解释成理想高斯机制的纯理论代价。')

page('八 配对统计与重复稳定性')
para('以下为本报告新增的探索性事后分析，未用于修改训练配置。先对每个项目的10个种子取均值，再以30个项目为配对单位。对五个非基线条件与无扰动基线的AUC、MCC、Recall和G-mean共20项双侧Wilcoxon符号秩检验，统一进行Holm校正。不同LOPO任务共享源项目，项目间独立性并不严格，因此p值应视为辅助证据。')
rows=[]
for v in a['contrasts']:
    if v['metric'] not in ['ROC_AUC','MCC']:continue
    rows.append([labels[v['condition']],v['metric'].replace('ROC_',''),f(v['delta']),f"[{v['ci95'][0]:.4f}, {v['ci95'][1]:.4f}]",f"{v['wins']}/{v['losses']}",f"{v['p_holm']:.4g}"])
table(['条件','指标','均值差','差值95%区间','胜／负','校正p'],rows,10)
para('表6　差值为当前条件减去无扰动。区间为对30个项目差值进行20000次配对bootstrap所得百分位区间，随机种子20261123；区间未作同时覆盖校正，与Wilcoxon检验的统计对象也不同。胜负按项目均值计数，本表均无平局。')
para('σ＝0.1的AUC胜出20个项目、MCC胜出17个项目，但两项差值区间均跨0，Holm校正p分别为0.7190和1.0000，不能声称具有统计显著提升。σ≥0.25的AUC和MCC下降在该探索性校正检验中均有支持。')
table(['条件','AUC 平均种子SD','MCC 平均种子SD','Recall 平均种子SD'],[[labels[c]]+[f(a['overall'][c][k]['mean_seed_sd']) for k in ['ROC_AUC','MCC','Recall']] for c in cs],10)
para('表7　先在每个项目内计算10次重复的样本标准差，再对30个项目平均。这里的SD描述随机种子稳定性，不是总体均值的标准误或置信区间。高噪声尤其增加MCC和Recall的重复波动。')

page('九 有效性限制与复现资料')
para('隐私有效性：未核算端到端ε/δ，辅助统计量未被完整保护；无攻击实验。因此本报告支持扰动—效用分析，不支持完整隐私保证或真实攻击风险排序。')
para('实现有效性：保留作者近似Focal梯度和部分对齐反向传播实现。高噪声条件出现明显数值饱和；未修改Sigmoid或其他算子来改善结果，以保持本批次实现一致。若后续采用稳定数值实现，应建立新版本并配套重跑对照，不能替换本批次个别结果。')
para('数据与评价有效性：源验证与其他源客户端的训练记录可能具有相同特征；目标无标签统计量参与预处理；不同家族特征和客户端数量不同；13个有效LOC项目的工作量结果不可直接推广至全部30个项目。阈值搜索范围固定，高噪声下大量边界取值需结合概率饱和解释。')
para('外部有效性：仅研究当前冻结30个项目、单一模型配置及四档非零噪声。σ＝0.1只是该网格内的最高平均分，不能称为全局最优，也不能再用目标测试分数选择部署参数。后续若细化0.1附近的网格，应预先固定设计并使用源验证或独立任务选择参数。')
para('复现入口：完整实验目录位于 E:/paper/outputs/fedcpdp-privacy-server-results-20261120/fedcpdp-privacy-native-20261120。study.json定义矩阵；conditions/<条件>/protocol.json定义参数；tasks.json和splits保存项目及行号划分；data_manifest.json记录数据哈希。')
para('结果入口：conditions/<条件>/runs/<目标>/<种子>/保存metrics.json、predictions.csv、validation_*.csv、features.json、console.log和checkpoint.pkl。各条件目录中的all_runs.csv、project_summary.csv和family_summary.csv为服务器复核后汇总；其中condition字段表示阈值口径，跨目录合并时还需单独保留隐私条件字段。')
para('本报告分析材料：同目录报告文件夹中的analysis.json保存全部重算结果和统计值；analyze_privacy_report.py、plot_privacy_report.py及build_privacy_report.py提供分析和文档生成步骤。基础仓库提交为 e99aacda94aed722e714dae249597de6429aac40；本次新增扫描包不应被误称为该提交已包含的功能。')
para('完整归档SHA-256：e0db6281c156aa264e5e5c8fa491b0fac7df09d4c3972f6ddaca30201e777141。作者核心文件SHA-256：dc8dba706c475e74f22fd7d68f513e7eae19e9bcf8007db9a0b4f57bed4f3687。')
para('依据材料：①本批次study.json与六套protocol.json；②作者vendor/fed_cpdp.py及本批次train.py、data.py、metrics.py；③1800份metrics.json与预测、日志；④各条件verification.json和本地download_verification.json；⑤论文 Federated and Privacy-Preserving Cross-Project Defect Prediction With Heterogeneous Features。论文参数归属沿用冻结协议的核对记录，本报告不声称严格复现原论文数值。')

for metric,title in [('ROC_AUC','附录一 各项目 AUC 均值'),('MCC','附录二 各项目 MCC 均值')]:
    page(title);para('每个单元格为该项目10个训练种子的平均值。项目内标准差与其他指标详见analysis.json。')
    table(['目标项目']+names,[[t]+[f(a['projects'][c][t][metric]['mean']) for c in cs] for t in sorted(m,key=lambda t:(m[t]['family'],t))],10)
    para('项目按家族排序：AEEEM、NASA、PROMISE、Relink、SOFTLAB。所有条件使用相同目标项目和冻结源端划分。')

for root in [d._element,d.styles.element]:
    for border in list(root.iter(qn('w:pBdr'))):border.getparent().remove(border)
d.save(O/'FedCPDP隐私扰动与预测效用实验报告.docx')
print(O/'FedCPDP隐私扰动与预测效用实验报告.docx')
