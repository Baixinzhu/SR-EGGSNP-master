import torch
import torch.nn as nn



class EGSNPCell(nn.Module):
    #input_size输入的特征维度，对应每个时间步的特征数，hidden_size为神经元个数
    def __init__(self, input_size, hidden_size,dropout_rate=0.0):
        super(EGSNPCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate

        # 输入门
        self.W_ir = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_hr = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.W_ur = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_ir = nn.Parameter(torch.Tensor(hidden_size))
        self.b_hr = nn.Parameter(torch.Tensor(hidden_size))
        self.b_ur = nn.Parameter(torch.Tensor(hidden_size))


        # 消耗门
        self.W_ic = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_hc = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.W_uc = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_ic = nn.Parameter(torch.Tensor(hidden_size))
        self.b_hc = nn.Parameter(torch.Tensor(hidden_size))
        self.b_uc = nn.Parameter(torch.Tensor(hidden_size))

        # 候选单元状态
        self.W_ia = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_ua = nn.Parameter(torch.Tensor(hidden_size, hidden_size))

        self.b_ia = nn.Parameter(torch.Tensor(hidden_size))
        self.b_ua = nn.Parameter(torch.Tensor(hidden_size))
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    def forward(self, x, init_states=None):
        #对Pytorch张量x的大小进行解包，x.size()返回x的大小。以元组的形式
        bs, _ = x.size()

        h_t, u_t = (torch.zeros(self.hidden_size).to(x.device),
                    torch.zeros(self.hidden_size).to(x.device)) if init_states is None else init_states

        # 输入门
        r_t = torch.sigmoid(x @ self.W_ir.t() + self.b_ir + h_t @ self.W_hr.t() + self.b_hr + u_t @ self.W_ur + self.b_ur)

        # 消耗门
        c_t = torch.sigmoid(x @ self.W_ic.t() + self.b_ic + h_t @ self.W_hc.t() + self.b_hc + u_t @ self.W_uc + self.b_uc)

        #候选状态（脉冲更新公式）
        a = torch.tanh(x @ self.W_ia.t() + self.b_ia + u_t @ self.W_ua.t() + self.b_ua)
        # 更新内部神经元状态

        u_t = r_t * u_t - c_t * a
        # 更新隐藏状态
        h_t = a
        h_t = self.dropout(h_t)
        return h_t, u_t
class EGSNP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,dropout_rate=0.0):
        super(EGSNP, self).__init__()
        self.EGSNP_cell = EGSNPCell(input_size, hidden_size,dropout_rate)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        bs, seq_len,_ = x.size()
        h_t, u_t = torch.zeros(self.EGSNP_cell.hidden_size).to(x.device), torch.zeros(self.EGSNP_cell.hidden_size).to(x.device)

        for t in range(seq_len):
            h_t, u_t = self.EGSNP_cell(x[:, t , :], (h_t, u_t))

        out = self.linear(h_t)
        return out
