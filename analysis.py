import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

data = {
    'Race': ['Melbourne', 'Shanghai', 'Suzuka', 'Sakhir', 'Jeddah', 'Miami', 
             'Imola', 'Monaco', 'Barcelona', 'Montreal', 'Spielberg', 'Silverstone',
             'Spa', 'Budapest', 'Zandvoort', 'Monza', 'Baku', 'Singapore', 
             'Austin', 'Mexico City', 'Sao Paulo'],
    'Country': ['Australia', 'China', 'Japan', 'Bahrain', 'Saudi Arabia', 'United States',
                'Italy', 'Monaco', 'Spain', 'Canada', 'Austria', 'Great Britain',
                'Belgium', 'Hungary', 'Netherlands', 'Italy', 'Azerbaijan', 'Singapore',
                'United States', 'Mexico', 'Brazil'],
    'TiO2_Status': ['Permitted', 'Permitted', 'Permitted', 'Permitted', 'Permitted', 'Permitted',
                    'Banned', 'Banned', 'Banned', 'Permitted', 'Banned', 'Permitted',
                    'Banned', 'Banned', 'Banned', 'Banned', 'Permitted', 'Permitted',
                    'Permitted', 'Permitted', 'Permitted'],
    'Qualifying': [16, 8, 6, 4, 5, 3, 13, 15, 6, 4, 9, 7, 18, 15, 11, 7, 4, 4, 7, 6, 2],
    'Race': [4, 6, 6, 11, 6, 6, np.nan, 18, np.nan, 3, np.nan, np.nan, 16, 10, 16, 9, 4, 5, 13, 6, 2]
}

df = pd.DataFrame(data)

df['DNF'] = df['Race'].isna()
df['Race_Clean'] = df['Race'].copy()

permitted = df[df['TiO2_Status'] == 'Permitted']
banned = df[df['TiO2_Status'] == 'Banned']

print("=" * 80)
print("KIMI ANTONELLI PERFORMANCE vs TITANIUM DIOXIDE REGULATIONS")
print("=" * 80)
print()

print("SAMPLE SIZES")
print(f"Races in TiO2-Permitted Countries: {len(permitted)}")
print(f"Races in TiO2-Banned Countries: {len(banned)}")
print()

print("=" * 80)
print("QUALIFYING ANALYSIS")
print("=" * 80)

perm_qual = permitted['Qualifying']
ban_qual = banned['Qualifying']

print(f"\nTiO2 PERMITTED - Mean Qualifying: {perm_qual.mean():.2f} (SD: {perm_qual.std():.2f})")
print(f"TiO2 BANNED - Mean Qualifying: {ban_qual.mean():.2f} (SD: {ban_qual.std():.2f})")

t_stat_qual, p_val_qual = stats.ttest_ind(perm_qual, ban_qual)
cohens_d_qual = (perm_qual.mean() - ban_qual.mean()) / np.sqrt(((len(perm_qual)-1)*perm_qual.std()**2 + (len(ban_qual)-1)*ban_qual.std()**2) / (len(perm_qual) + len(ban_qual) - 2))

print(f"\nt-test: t = {t_stat_qual:.3f}, p = {p_val_qual:.4f}")
print(f"Cohen's d = {cohens_d_qual:.3f}")
print()

print("=" * 80)
print("RACE FINISH ANALYSIS (Excluding DNFs)")
print("=" * 80)

perm_race = permitted['Race_Clean'].dropna()
ban_race = banned['Race_Clean'].dropna()

print(f"\nTiO2 PERMITTED - Mean Race Position: {perm_race.mean():.2f} (SD: {perm_race.std():.2f}) [n={len(perm_race)}]")
print(f"TiO2 BANNED - Mean Race Position: {ban_race.mean():.2f} (SD: {ban_race.std():.2f}) [n={len(ban_race)}]")

t_stat_race, p_val_race = stats.ttest_ind(perm_race, ban_race)
cohens_d_race = (perm_race.mean() - ban_race.mean()) / np.sqrt(((len(perm_race)-1)*perm_race.std()**2 + (len(ban_race)-1)*ban_race.std()**2) / (len(perm_race) + len(ban_race) - 2))

print(f"\nt-test: t = {t_stat_race:.3f}, p = {p_val_race:.4f}")
print(f"Cohen's d = {cohens_d_race:.3f}")
print()

print("=" * 80)
print("DNF ANALYSIS")
print("=" * 80)

perm_dnf_rate = permitted['DNF'].sum() / len(permitted)
ban_dnf_rate = banned['DNF'].sum() / len(banned)

print(f"\nTiO2 PERMITTED - DNF Rate: {perm_dnf_rate:.1%} ({permitted['DNF'].sum()}/{len(permitted)})")
print(f"TiO2 BANNED - DNF Rate: {ban_dnf_rate:.1%} ({banned['DNF'].sum()}/{len(banned)})")

contingency_table = pd.crosstab(df['TiO2_Status'], df['DNF'])
chi2, p_val_dnf, dof, expected = stats.chi2_contingency(contingency_table)

print(f"\nChi-square test: χ² = {chi2:.3f}, p = {p_val_dnf:.4f}")
print()

sns.set_style("whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

colors = {'Permitted': '#00D2BE', 'Banned': '#E10600'}

ax1 = axes[0, 0]
df_plot = df.copy()
df_plot['TiO2_Status'] = pd.Categorical(df_plot['TiO2_Status'], categories=['Permitted', 'Banned'])
sns.boxplot(data=df_plot, x='TiO2_Status', y='Qualifying', ax=ax1, palette=colors)
sns.stripplot(data=df_plot, x='TiO2_Status', y='Qualifying', ax=ax1, color='black', alpha=0.5, size=6)
ax1.set_ylabel('Qualifying Position', fontsize=12, fontweight='bold')
ax1.set_xlabel('Titanium Dioxide Status', fontsize=12, fontweight='bold')
ax1.set_title('Qualifying Performance', fontsize=14, fontweight='bold')
ax1.invert_yaxis()

ax2 = axes[0, 1]
df_race_clean = df.dropna(subset=['Race_Clean']).copy()
df_race_clean['TiO2_Status'] = pd.Categorical(df_race_clean['TiO2_Status'], categories=['Permitted', 'Banned'])
sns.boxplot(data=df_race_clean, x='TiO2_Status', y='Race_Clean', ax=ax2, palette=colors)
sns.stripplot(data=df_race_clean, x='TiO2_Status', y='Race_Clean', ax=ax2, color='black', alpha=0.5, size=6)
ax2.set_ylabel('Race Finish Position', fontsize=12, fontweight='bold')
ax2.set_xlabel('Titanium Dioxide Status', fontsize=12, fontweight='bold')
ax2.set_title('Race Performance (Excl. DNFs)', fontsize=14, fontweight='bold')
ax2.invert_yaxis()

ax3 = axes[1, 0]
dnf_data = pd.DataFrame({
    'Status': ['Permitted', 'Banned'],
    'DNF_Rate': [perm_dnf_rate * 100, ban_dnf_rate * 100]
})
bars = ax3.bar(dnf_data['Status'], dnf_data['DNF_Rate'], color=[colors['Permitted'], colors['Banned']], edgecolor='black', linewidth=1.5)
ax3.set_ylabel('DNF Rate (%)', fontsize=12, fontweight='bold')
ax3.set_xlabel('Titanium Dioxide Status', fontsize=12, fontweight='bold')
ax3.set_title('Did Not Finish Rate', fontsize=14, fontweight='bold')
for bar in bars:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

ax4 = axes[1, 1]
summary_data = pd.DataFrame({
    'Metric': ['Avg Qualifying', 'Avg Race Finish', 'DNF Rate (%)'],
    'Permitted': [perm_qual.mean(), perm_race.mean(), perm_dnf_rate * 100],
    'Banned': [ban_qual.mean(), ban_race.mean(), ban_dnf_rate * 100]
})
x = np.arange(len(summary_data['Metric']))
width = 0.35
bars1 = ax4.bar(x - width/2, summary_data['Permitted'], width, label='Permitted', color=colors['Permitted'], edgecolor='black', linewidth=1.5)
bars2 = ax4.bar(x + width/2, summary_data['Banned'], width, label='Banned', color=colors['Banned'], edgecolor='black', linewidth=1.5)
ax4.set_ylabel('Value', fontsize=12, fontweight='bold')
ax4.set_title('Summary Comparison', fontsize=14, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(summary_data['Metric'], fontsize=10)
ax4.legend(title='TiO2 Status', fontsize=10)

plt.tight_layout()
plt.savefig(r'C:\Users\jessm\OneDrive\Desktop\F1 Notebooks\2025 Season\Antonelli\antonelli_tio2_analysis.png', dpi=300, bbox_inches='tight')
print("Visualization saved")

fig2, ax = plt.subplots(figsize=(12, 8))
for status in ['Permitted', 'Banned']:
    subset = df[df['TiO2_Status'] == status].dropna(subset=['Race_Clean'])
    ax.scatter(subset['Qualifying'], subset['Race_Clean'], 
              label=f'TiO2 {status}', color=colors[status], s=150, alpha=0.7, edgecolors='black', linewidth=1.5)

ax.plot([0, 20], [0, 20], 'k--', alpha=0.3, linewidth=1)
ax.set_xlabel('Qualifying Position', fontsize=14, fontweight='bold')
ax.set_ylabel('Race Finish Position', fontsize=14, fontweight='bold')
ax.set_title('Qualifying vs Race Performance by TiO2 Regulation Status', fontsize=16, fontweight='bold')
ax.legend(fontsize=12)
ax.invert_xaxis()
ax.invert_yaxis()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(r'C:\Users\jessm\OneDrive\Desktop\F1 Notebooks\2025 Season\Antonelli\antonelli_tio2_scatter.png', dpi=300, bbox_inches='tight')
print("Scatter plot saved ")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)