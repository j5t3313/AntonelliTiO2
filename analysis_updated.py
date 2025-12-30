import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.facecolor'] = '#1a1a1a'
plt.rcParams['figure.facecolor'] = '#0d0d0d'
plt.rcParams['text.color'] = 'white'
plt.rcParams['axes.labelcolor'] = 'white'
plt.rcParams['xtick.color'] = 'white'
plt.rcParams['ytick.color'] = 'white'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['grid.color'] = '#333333'

antonelli_data = {
    'Race': ['Melbourne', 'Shanghai', 'Suzuka', 'Sakhir', 'Jeddah', 'Miami', 
             'Imola', 'Monaco', 'Barcelona', 'Montreal', 'Spielberg', 'Silverstone',
             'Spa', 'Budapest', 'Zandvoort', 'Monza', 'Baku', 'Singapore', 
             'Austin', 'Mexico City', 'Sao Paulo', 'Las Vegas', 'Qatar', 'Abu Dhabi'],
    'Country': ['Australia', 'China', 'Japan', 'Bahrain', 'Saudi Arabia', 'United States',
                'Italy', 'Monaco', 'Spain', 'Canada', 'Austria', 'Great Britain',
                'Belgium', 'Hungary', 'Netherlands', 'Italy', 'Azerbaijan', 'Singapore',
                'United States', 'Mexico', 'Brazil', 'United States', 'Qatar', 'United Arab Emirates'],
    'TiO2_Status': ['Permitted', 'Permitted', 'Permitted', 'Permitted', 'Permitted', 'Permitted',
                    'Banned', 'Banned', 'Banned', 'Permitted', 'Banned', 'Permitted',
                    'Banned', 'Banned', 'Banned', 'Banned', 'Permitted', 'Permitted',
                    'Permitted', 'Permitted', 'Permitted', 'Permitted', 'Banned', 'Banned'],
    'Qualifying': [16, 8, 6, 4, 5, 3, 13, 15, 6, 4, 9, 7, 18, 15, 11, 7, 4, 4, 7, 6, 2, 17, 5, 14],
    'Race_Finish': [4, 6, 6, 11, 6, 6, np.nan, 18, np.nan, 3, np.nan, np.nan, 16, 10, 16, 9, 4, 5, 13, 6, 2, 3, 5, 15]
}

russell_data = {
    'Race': ['Melbourne', 'Shanghai', 'Suzuka', 'Sakhir', 'Jeddah', 'Miami', 
             'Imola', 'Monaco', 'Barcelona', 'Montreal', 'Spielberg', 'Silverstone',
             'Spa', 'Budapest', 'Zandvoort', 'Monza', 'Baku', 'Singapore', 
             'Austin', 'Mexico City', 'Sao Paulo', 'Las Vegas', 'Qatar', 'Abu Dhabi'],
    'TiO2_Status': ['Permitted', 'Permitted', 'Permitted', 'Permitted', 'Permitted', 'Permitted',
                    'Banned', 'Banned', 'Banned', 'Permitted', 'Banned', 'Permitted',
                    'Banned', 'Banned', 'Banned', 'Banned', 'Permitted', 'Permitted',
                    'Permitted', 'Permitted', 'Permitted', 'Permitted', 'Banned', 'Banned'],
    'Qualifying': [4, 3, 3, 3, 5, 5, 3, 14, 4, 1, 5, 4, 6, 4, 5, 5, 5, 1, 4, 4, 6, 4, 4, 4],
    'Race_Finish': [3, 2, 5, 2, 3, 3, 7, 11, 4, 1, 5, 10, 5, 3, 4, 5, 2, 1, 6, 7, 4, 2, 6, 5]
}

df_ant = pd.DataFrame(antonelli_data)
df_rus = pd.DataFrame(russell_data)

df_ant['DNF'] = df_ant['Race_Finish'].isna()
df_ant['Race_Clean'] = df_ant['Race_Finish'].copy()
df_rus['DNF'] = df_rus['Race_Finish'].isna()
df_rus['Race_Clean'] = df_rus['Race_Finish'].copy()

def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    pooled_std = np.sqrt(((n1-1)*group1.std()**2 + (n2-1)*group2.std()**2) / (n1 + n2 - 2))
    return (group1.mean() - group2.mean()) / pooled_std

def bootstrap_cohens_d_ci(group1, group2, n_bootstrap=10000, ci=95):
    np.random.seed(42)
    d_values = []
    g1 = np.array(group1)
    g2 = np.array(group2)
    for _ in range(n_bootstrap):
        sample1 = np.random.choice(g1, size=len(g1), replace=True)
        sample2 = np.random.choice(g2, size=len(g2), replace=True)
        n1, n2 = len(sample1), len(sample2)
        pooled_std = np.sqrt(((n1-1)*sample1.std()**2 + (n2-1)*sample2.std()**2) / (n1 + n2 - 2))
        if pooled_std > 0:
            d = (sample1.mean() - sample2.mean()) / pooled_std
            d_values.append(d)
    lower = np.percentile(d_values, (100 - ci) / 2)
    upper = np.percentile(d_values, 100 - (100 - ci) / 2)
    return lower, upper

ant_permitted = df_ant[df_ant['TiO2_Status'] == 'Permitted']
ant_banned = df_ant[df_ant['TiO2_Status'] == 'Banned']
rus_permitted = df_rus[df_rus['TiO2_Status'] == 'Permitted']
rus_banned = df_rus[df_rus['TiO2_Status'] == 'Banned']

print("=" * 80)
print("KIMI ANTONELLI PERFORMANCE vs TITANIUM DIOXIDE REGULATIONS")
print("=" * 80)
print()

print("SAMPLE SIZES")
print(f"Races in TiO2-Permitted Countries: {len(ant_permitted)}")
print(f"Races in TiO2-Banned Countries: {len(ant_banned)}")
print()

print("=" * 80)
print("QUALIFYING ANALYSIS")
print("=" * 80)

ant_perm_qual = ant_permitted['Qualifying']
ant_ban_qual = ant_banned['Qualifying']

print(f"\nTiO2 PERMITTED - Mean Qualifying: {ant_perm_qual.mean():.2f} (SD: {ant_perm_qual.std():.2f})")
print(f"TiO2 BANNED - Mean Qualifying: {ant_ban_qual.mean():.2f} (SD: {ant_ban_qual.std():.2f})")

t_stat_qual, p_val_qual = stats.ttest_ind(ant_perm_qual, ant_ban_qual)
d_qual = cohens_d(ant_perm_qual, ant_ban_qual)
d_qual_ci = bootstrap_cohens_d_ci(ant_perm_qual, ant_ban_qual)

u_stat_qual, p_val_mw_qual = stats.mannwhitneyu(ant_perm_qual, ant_ban_qual, alternative='two-sided')

print(f"\nt-test: t = {t_stat_qual:.3f}, p = {p_val_qual:.4f}")
print(f"Mann-Whitney U: U = {u_stat_qual:.1f}, p = {p_val_mw_qual:.4f}")
print(f"Cohen's d = {d_qual:.3f} [95% CI: {d_qual_ci[0]:.3f}, {d_qual_ci[1]:.3f}]")
print()

print("=" * 80)
print("RACE FINISH ANALYSIS (Excluding DNFs)")
print("=" * 80)

ant_perm_race = ant_permitted['Race_Clean'].dropna()
ant_ban_race = ant_banned['Race_Clean'].dropna()

print(f"\nTiO2 PERMITTED - Mean Race Position: {ant_perm_race.mean():.2f} (SD: {ant_perm_race.std():.2f}) [n={len(ant_perm_race)}]")
print(f"TiO2 BANNED - Mean Race Position: {ant_ban_race.mean():.2f} (SD: {ant_ban_race.std():.2f}) [n={len(ant_ban_race)}]")

t_stat_race, p_val_race = stats.ttest_ind(ant_perm_race, ant_ban_race)
d_race = cohens_d(ant_perm_race, ant_ban_race)
d_race_ci = bootstrap_cohens_d_ci(ant_perm_race, ant_ban_race)

u_stat_race, p_val_mw_race = stats.mannwhitneyu(ant_perm_race, ant_ban_race, alternative='two-sided')

print(f"\nt-test: t = {t_stat_race:.3f}, p = {p_val_race:.4f}")
print(f"Mann-Whitney U: U = {u_stat_race:.1f}, p = {p_val_mw_race:.4f}")
print(f"Cohen's d = {d_race:.3f} [95% CI: {d_race_ci[0]:.3f}, {d_race_ci[1]:.3f}]")
print()

print("=" * 80)
print("DNF ANALYSIS")
print("=" * 80)

ant_perm_dnf_rate = ant_permitted['DNF'].sum() / len(ant_permitted)
ant_ban_dnf_rate = ant_banned['DNF'].sum() / len(ant_banned)

print(f"\nTiO2 PERMITTED - DNF Rate: {ant_perm_dnf_rate:.1%} ({ant_permitted['DNF'].sum()}/{len(ant_permitted)})")
print(f"TiO2 BANNED - DNF Rate: {ant_ban_dnf_rate:.1%} ({ant_banned['DNF'].sum()}/{len(ant_banned)})")

contingency_table = pd.crosstab(df_ant['TiO2_Status'], df_ant['DNF'])
chi2, p_val_dnf, dof, expected = stats.chi2_contingency(contingency_table)

print(f"\nChi-square test: χ² = {chi2:.3f}, p = {p_val_dnf:.4f}")
print()

print("=" * 80)
print("TEAMMATE COMPARISON: GEORGE RUSSELL")
print("=" * 80)

rus_perm_qual = rus_permitted['Qualifying']
rus_ban_qual = rus_banned['Qualifying']
rus_perm_race = rus_permitted['Race_Clean'].dropna()
rus_ban_race = rus_banned['Race_Clean'].dropna()

print("\nQUALIFYING:")
print(f"TiO2 PERMITTED - Mean: {rus_perm_qual.mean():.2f} (SD: {rus_perm_qual.std():.2f})")
print(f"TiO2 BANNED - Mean: {rus_ban_qual.mean():.2f} (SD: {rus_ban_qual.std():.2f})")

t_rus_qual, p_rus_qual = stats.ttest_ind(rus_perm_qual, rus_ban_qual)
u_rus_qual, p_mw_rus_qual = stats.mannwhitneyu(rus_perm_qual, rus_ban_qual, alternative='two-sided')
d_rus_qual = cohens_d(rus_perm_qual, rus_ban_qual)
d_rus_qual_ci = bootstrap_cohens_d_ci(rus_perm_qual, rus_ban_qual)

print(f"t-test: t = {t_rus_qual:.3f}, p = {p_rus_qual:.4f}")
print(f"Mann-Whitney U: U = {u_rus_qual:.1f}, p = {p_mw_rus_qual:.4f}")
print(f"Cohen's d = {d_rus_qual:.3f} [95% CI: {d_rus_qual_ci[0]:.3f}, {d_rus_qual_ci[1]:.3f}]")

print("\nRACE FINISH:")
print(f"TiO2 PERMITTED - Mean: {rus_perm_race.mean():.2f} (SD: {rus_perm_race.std():.2f})")
print(f"TiO2 BANNED - Mean: {rus_ban_race.mean():.2f} (SD: {rus_ban_race.std():.2f})")

t_rus_race, p_rus_race = stats.ttest_ind(rus_perm_race, rus_ban_race)
u_rus_race, p_mw_rus_race = stats.mannwhitneyu(rus_perm_race, rus_ban_race, alternative='two-sided')
d_rus_race = cohens_d(rus_perm_race, rus_ban_race)
d_rus_race_ci = bootstrap_cohens_d_ci(rus_perm_race, rus_ban_race)

print(f"t-test: t = {t_rus_race:.3f}, p = {p_rus_race:.4f}")
print(f"Mann-Whitney U: U = {u_rus_race:.1f}, p = {p_mw_rus_race:.4f}")
print(f"Cohen's d = {d_rus_race:.3f} [95% CI: {d_rus_race_ci[0]:.3f}, {d_rus_race_ci[1]:.3f}]")
print()

colors = {'Permitted': '#00D2BE', 'Banned': '#6c757d'}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

ax1 = axes[0, 0]
df_plot = df_ant.copy()
df_plot['TiO2_Status'] = pd.Categorical(df_plot['TiO2_Status'], categories=['Permitted', 'Banned'])
box1 = ax1.boxplot([ant_perm_qual, ant_ban_qual], positions=[1, 2], widths=0.6, patch_artist=True)
box1['boxes'][0].set_facecolor(colors['Permitted'])
box1['boxes'][1].set_facecolor(colors['Banned'])
for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
    for item in box1[element]:
        item.set_color('white')
ax1.scatter(np.ones(len(ant_perm_qual)) + np.random.normal(0, 0.04, len(ant_perm_qual)), ant_perm_qual, color='white', alpha=0.6, s=40, zorder=3)
ax1.scatter(np.ones(len(ant_ban_qual))*2 + np.random.normal(0, 0.04, len(ant_ban_qual)), ant_ban_qual, color='white', alpha=0.6, s=40, zorder=3)
ax1.set_xticks([1, 2])
ax1.set_xticklabels(['Permitted', 'Banned'])
ax1.set_ylabel('Qualifying Position', fontsize=12, fontweight='bold')
ax1.set_xlabel('Titanium Dioxide Status', fontsize=12, fontweight='bold')
ax1.set_title('Qualifying Performance', fontsize=14, fontweight='bold', color='#00D2BE')
ax1.invert_yaxis()

ax2 = axes[0, 1]
box2 = ax2.boxplot([ant_perm_race, ant_ban_race], positions=[1, 2], widths=0.6, patch_artist=True)
box2['boxes'][0].set_facecolor(colors['Permitted'])
box2['boxes'][1].set_facecolor(colors['Banned'])
for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
    for item in box2[element]:
        item.set_color('white')
ax2.scatter(np.ones(len(ant_perm_race)) + np.random.normal(0, 0.04, len(ant_perm_race)), ant_perm_race, color='white', alpha=0.6, s=40, zorder=3)
ax2.scatter(np.ones(len(ant_ban_race))*2 + np.random.normal(0, 0.04, len(ant_ban_race)), ant_ban_race, color='white', alpha=0.6, s=40, zorder=3)
ax2.set_xticks([1, 2])
ax2.set_xticklabels(['Permitted', 'Banned'])
ax2.set_ylabel('Race Finish Position', fontsize=12, fontweight='bold')
ax2.set_xlabel('Titanium Dioxide Status', fontsize=12, fontweight='bold')
ax2.set_title('Race Performance (Excl. DNFs)', fontsize=14, fontweight='bold', color='#00D2BE')
ax2.invert_yaxis()

ax3 = axes[1, 0]
dnf_data = pd.DataFrame({
    'Status': ['Permitted', 'Banned'],
    'DNF_Rate': [ant_perm_dnf_rate * 100, ant_ban_dnf_rate * 100]
})
bars = ax3.bar(dnf_data['Status'], dnf_data['DNF_Rate'], color=[colors['Permitted'], colors['Banned']], edgecolor='white', linewidth=1.5)
ax3.set_ylabel('DNF Rate (%)', fontsize=12, fontweight='bold')
ax3.set_xlabel('Titanium Dioxide Status', fontsize=12, fontweight='bold')
ax3.set_title('Did Not Finish Rate', fontsize=14, fontweight='bold', color='#00D2BE')
for bar in bars:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
            f'{height:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold', color='white')

ax4 = axes[1, 1]
summary_data = pd.DataFrame({
    'Metric': ['Avg Qualifying', 'Avg Race Finish', 'DNF Rate (%)'],
    'Permitted': [ant_perm_qual.mean(), ant_perm_race.mean(), ant_perm_dnf_rate * 100],
    'Banned': [ant_ban_qual.mean(), ant_ban_race.mean(), ant_ban_dnf_rate * 100]
})
x = np.arange(len(summary_data['Metric']))
width = 0.35
bars1 = ax4.bar(x - width/2, summary_data['Permitted'], width, label='Permitted', color=colors['Permitted'], edgecolor='white', linewidth=1.5)
bars2 = ax4.bar(x + width/2, summary_data['Banned'], width, label='Banned', color=colors['Banned'], edgecolor='white', linewidth=1.5)
ax4.set_ylabel('Value', fontsize=12, fontweight='bold')
ax4.set_title('Summary Comparison', fontsize=14, fontweight='bold', color='#00D2BE')
ax4.set_xticks(x)
ax4.set_xticklabels(summary_data['Metric'], fontsize=10)
ax4.legend(title='TiO2 Status', fontsize=10, facecolor='#1a1a1a', edgecolor='white', labelcolor='white', title_fontsize=10)
ax4.get_legend().get_title().set_color('white')

plt.tight_layout()
plt.savefig('antonelli_tio2_analysis.png', dpi=300, bbox_inches='tight', facecolor='#0d0d0d')
print("Main visualization saved")

fig2, ax = plt.subplots(figsize=(12, 8))
for status, color in colors.items():
    subset = df_ant[df_ant['TiO2_Status'] == status].dropna(subset=['Race_Clean'])
    ax.scatter(subset['Qualifying'], subset['Race_Clean'], 
              label=f'TiO2 {status}', color=color, s=150, alpha=0.8, edgecolors='white', linewidth=1.5)

ax.plot([0, 20], [0, 20], '--', color='#00D2BE', alpha=0.4, linewidth=1)
ax.set_xlabel('Qualifying Position', fontsize=14, fontweight='bold')
ax.set_ylabel('Race Finish Position', fontsize=14, fontweight='bold')
ax.set_title('Qualifying vs Race Performance by TiO2 Regulation Status', fontsize=16, fontweight='bold', color='#00D2BE')
ax.legend(fontsize=12, facecolor='#1a1a1a', edgecolor='white', labelcolor='white')
ax.invert_xaxis()
ax.invert_yaxis()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('antonelli_tio2_scatter.png', dpi=300, bbox_inches='tight', facecolor='#0d0d0d')
print("Scatter plot saved")

fig3, axes = plt.subplots(1, 2, figsize=(14, 6))

driver_colors = {'Antonelli': '#00D2BE', 'Russell': '#C0C0C0'}

ax1 = axes[0]
x_pos = np.array([0, 1, 2.5, 3.5])
qual_effects = [d_qual, d_rus_qual]
qual_ci_lows = [d_qual_ci[0], d_rus_qual_ci[0]]
qual_ci_highs = [d_qual_ci[1], d_rus_qual_ci[1]]
race_effects = [d_race, d_rus_race]
race_ci_lows = [d_race_ci[0], d_rus_race_ci[0]]
race_ci_highs = [d_race_ci[1], d_rus_race_ci[1]]

all_effects = qual_effects + race_effects
all_ci_lows = qual_ci_lows + race_ci_lows
all_ci_highs = qual_ci_highs + race_ci_highs
all_colors = [driver_colors['Antonelli'], driver_colors['Russell'], driver_colors['Antonelli'], driver_colors['Russell']]

bars = ax1.bar(x_pos, all_effects, color=all_colors, edgecolor='white', linewidth=1.5, width=0.8)
errors_low = [e - l for e, l in zip(all_effects, all_ci_lows)]
errors_high = [h - e for e, h in zip(all_effects, all_ci_highs)]
ax1.errorbar(x_pos, all_effects, yerr=[errors_low, errors_high], fmt='none', color='white', capsize=5, capthick=2, linewidth=2)
ax1.axhline(y=0, color='white', linestyle='-', linewidth=0.5, alpha=0.5)
ax1.axhline(y=-0.8, color='#00D2BE', linestyle='--', linewidth=1, alpha=0.5)
ax1.axhline(y=0.8, color='#00D2BE', linestyle='--', linewidth=1, alpha=0.5)
ax1.set_xticks(x_pos)
ax1.set_xticklabels(['ANT\nQualifying', 'RUS\nQualifying', 'ANT\nRace', 'RUS\nRace'])
ax1.set_ylabel("Cohen's d (Permitted - Banned)", fontsize=12, fontweight='bold')
ax1.set_title("Effect Size Comparison: TiO2 Status Impact\n(with 95% Bootstrap CI)", fontsize=14, fontweight='bold', color='#00D2BE')
ax1.text(3.9, -0.8, 'Large effect\nthreshold', fontsize=9, color='#00D2BE', alpha=0.7, va='center')

ax2 = axes[1]
width = 0.35
x = np.array([0, 1])

ant_means = [ant_perm_qual.mean(), ant_perm_race.mean()]
rus_means = [rus_perm_qual.mean(), rus_perm_race.mean()]
ant_banned_means = [ant_ban_qual.mean(), ant_ban_race.mean()]
rus_banned_means = [rus_ban_qual.mean(), rus_ban_race.mean()]

x_offset = np.array([0, 1.5])
ax2.bar(x_offset - 0.3, ant_means, width, label='Antonelli - Permitted', color=driver_colors['Antonelli'], edgecolor='white', linewidth=1.5)
ax2.bar(x_offset + 0.0, [ant_ban_qual.mean(), ant_ban_race.mean()], width, label='Antonelli - Banned', color=driver_colors['Antonelli'], edgecolor='white', linewidth=1.5, alpha=0.5)
ax2.bar(x_offset + 0.3, rus_means, width, label='Russell - Permitted', color=driver_colors['Russell'], edgecolor='white', linewidth=1.5)
ax2.bar(x_offset + 0.6, rus_banned_means, width, label='Russell - Banned', color=driver_colors['Russell'], edgecolor='white', linewidth=1.5, alpha=0.5)

ax2.set_xticks(x_offset + 0.15)
ax2.set_xticklabels(['Qualifying', 'Race Finish'])
ax2.set_ylabel('Average Position', fontsize=12, fontweight='bold')
ax2.set_title('Teammate Comparison: Mean Positions by TiO2 Status', fontsize=14, fontweight='bold', color='#00D2BE')
ax2.legend(fontsize=9, facecolor='#1a1a1a', edgecolor='white', labelcolor='white', loc='upper left')
ax2.invert_yaxis()

plt.tight_layout()
plt.savefig('antonelli_teammate_comparison.png', dpi=300, bbox_inches='tight', facecolor='#0d0d0d')
print("Teammate comparison saved")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)