import numpy as np
import pandas as pd
import os
import glob
from app import get_altum_age, get_grim_age, get_horvath_age, get_hannum_age, get_horvath_sb_age, get_pheno_age

def get_ensemble_age(df):
    return np.mean([get_altum_age(df), get_horvath_age(df), get_hannum_age(df), get_pheno_age(df), get_horvath_sb_age(df), get_grim_age(df)], axis=0)

def get_true_age_labels(file):
    age_data = pd.read_table(file).iloc[9,1:]
    ages = []
    for i in range(len(age_data)):
        ages.append(int(age_data[i].split('-')[0])+5)
    gender_data = pd.read_table(file).iloc[6,1:]
    genders = []
    for i in range(len(gender_data)):
        genders.append(int(gender_data[i]) - 1)
    return np.array(ages), np.array(genders)

missing_altum_cpgs = ['cg00149659', 'cg00155167', 'cg00210842', 'cg00214855', 'cg00280814', 'cg00309204', 'cg00398048', 'cg00432979', 'cg00436282', 'cg00461841', 'cg00547018', 'cg00551244', 'cg00630583', 'cg00648153', 'cg00650762', 'cg00659129', 'cg00917893', 'cg01036173', 'cg01139966', 'cg01311051', 'cg01371477', 'cg01381846', 'cg01481441', 'cg01488147', 'cg01491225', 'cg01511567', 'cg01533387', 'cg01578341', 'cg01630869', 'cg01675895', 'cg01725199', 'cg01773854', 'cg01776246', 'cg01808130', 'cg01813965', 'cg01817393', 'cg01889448', 'cg01936270', 'cg01990334', 'cg01994779', 'cg02065387', 'cg02105377', 'cg02121943', 'cg02284889', 'cg02304930', 'cg02309273', 'cg02342494', 'cg02430692', 'cg02475653', 'cg02600515', 'cg02646854', 'cg02655623', 'cg02724472', 'cg02756614', 'cg02757432', 'cg02831393', 'cg02888247', 'cg02904235', 'cg02916816', 'cg03085637', 'cg03102516', 'cg03148461', 'cg03165700', 'cg03221914', 'cg03264414', 'cg03302287', 'cg03382304', 'cg03454353', 'cg03476370', 'cg03600687', 'cg03684977', 'cg03747695', 'cg03750407', 'cg03785807', 'cg03807314', 'cg03909500', 'cg03955296', 'cg04001802', 'cg04033650', 'cg04063348', 'cg04117338', 'cg04121771', 'cg04126427', 'cg04187545', 'cg04219321', 'cg04229238', 'cg04338788', 'cg04368877', 'cg04376617', 'cg04497885', 'cg04599297', 'cg04616566', 'cg04619859', 'cg04705866', 'cg04743872', 'cg04752565', 'cg04762213', 'cg04820387', 'cg04856689', 'cg05087948', 'cg05130485', 'cg05164185', 'cg05168404', 'cg05189291', 'cg05200311', 'cg05205664', 'cg05313261', 'cg05321960', 'cg05380910', 'cg05411032', 'cg05467458', 'cg05473175', 'cg05564251', 'cg05669210', 'cg05726109', 'cg05727959', 'cg05779068', 'cg06007645', 'cg06148175', 'cg06150803', 'cg06151964', 'cg06251129', 'cg06291334', 'cg06321883', 'cg06356454', 'cg06366981', 'cg06386533', 'cg06459327', 'cg06491116', 'cg06506864', 'cg06537230', 'cg06665322', 'cg06718696', 'cg06725035', 'cg06885782', 'cg06995715', 'cg07009002', 'cg07207937', 'cg07274506', 'cg07295034', 'cg07398350', 'cg07414384', 'cg07503829', 'cg07613278', 'cg07707498', 'cg07844021', 'cg07903918', 'cg07928695', 'cg07979752', 'cg08089301', 'cg08126211', 'cg08198370', 'cg08212685', 'cg08228917', 'cg08426384', 'cg08521225', 'cg08576197', 'cg08578305', 'cg08596544', 'cg08661227', 'cg08662074', 'cg08674093', 'cg08724517', 'cg08797194', 'cg08859916', 'cg08927738', 'cg08935003', 'cg09009259', 'cg09087966', 'cg09205751', 'cg09296044', 'cg09338170', 'cg09374949', 'cg09375488', 'cg09386615', 'cg09453737', 'cg09559551', 'cg09638834', 'cg09688763', 'cg09752703', 'cg09868597', 'cg09871315', 'cg09872233', 'cg09929564', 'cg10000775', 'cg10021735', 'cg10025865', 'cg10058540', 'cg10146929', 'cg10247252', 'cg10305797', 'cg10365880', 'cg10367730', 'cg10409680', 'cg10450322', 'cg10453040', 'cg10773869', 'cg10878896', 'cg11114344', 'cg11126134', 'cg11189837', 'cg11337525', 'cg11388238', 'cg11405695', 'cg11484872', 'cg11593656', 'cg11608114', 'cg11654620', 'cg11655691', 'cg11670211', 'cg11781389', 'cg11877382', 'cg11911418', 'cg11913104', 'cg12067287', 'cg12126248', 'cg12187213', 'cg12188416', 'cg12188860', 'cg12274479', 'cg12288726', 'cg12365667', 'cg12368241', 'cg12403575', 'cg12503243', 'cg12513379', 'cg12535715', 'cg12542604', 'cg12556991', 'cg12578480', 'cg12624523', 'cg12627583', 'cg12629325', 'cg12638745', 'cg12643449', 'cg12644353', 'cg12646585', 'cg12696750', 'cg12737574', 'cg12758687', 'cg12830829', 'cg12835684', 'cg12884406', 'cg12914014', 'cg12952136', 'cg12991341', 'cg13058581', 'cg13080465', 'cg13131015', 'cg13234848', 'cg13243219', 'cg13297960', 'cg13326338', 'cg13372488', 'cg13410437', 'cg13633026', 'cg13634678', 'cg13645811', 'cg13654195', 'cg13722123', 'cg13735974', 'cg13877915', 'cg13882835', 'cg14047008', 'cg14091223', 'cg14133708', 'cg14138171', 'cg14149007', 'cg14178895', 'cg14236602', 'cg14318370', 'cg14329157', 'cg14372394', 'cg14423778', 'cg14426525', 'cg14472778', 'cg14545899', 'cg14613972', 'cg14671488', 'cg14700821', 'cg14795305', 'cg14800883', 'cg14802310', 'cg14839932', 'cg14916288', 'cg14927277', 'cg14932684', 'cg14948436', 'cg14981132', 'cg15078479', 'cg15214092', 'cg15269875', 'cg15308737', 'cg15316289', 'cg15329467', 'cg15350455', 'cg15411984', 'cg15481539', 'cg15488251', 'cg15565533', 'cg15572787', 'cg15597540', 'cg15605888', 'cg15673110', 'cg15687659', 'cg15707568', 'cg15727320', 'cg15736336', 'cg15739944', 'cg15747933', 'cg15792688', 'cg15798530', 'cg15824080', 'cg15869022', 'cg15903421', 'cg15977816', 'cg16001460', 'cg16007185', 'cg16094954', 'cg16121444', 'cg16173109', 'cg16185365', 'cg16250754', 'cg16257091', 'cg16280313', 'cg16310717', 'cg16341373', 'cg16361890', 'cg16427670', 'cg16551261', 'cg16592658', 'cg16677885', 'cg16689634', 'cg16721202', 'cg16779976', 'cg16796590', 'cg17133183', 'cg17304433', 'cg17304878', 'cg17352004', 'cg17353431', 'cg17416146', 'cg17543123', 'cg17563769', 'cg17582250', 'cg17607024', 'cg17683775', 'cg17701886', 'cg17729941', 'cg17754980', 'cg17851105', 'cg17890764', 'cg17895873', 'cg17904739', 'cg17920197', 'cg17971003', 'cg17990871', 'cg18003135', 'cg18139900', 'cg18190433', 'cg18202456', 'cg18219226', 'cg18248112', 'cg18292394', 'cg18357645', 'cg18384097', 'cg18392482', 'cg18413900', 'cg18427589', 'cg18587364', 'cg18619831', 'cg18641050', 'cg18704595', 'cg18722841', 'cg18811423', 'cg18953280', 'cg19072037', 'cg19154173', 'cg19167673', 'cg19356324', 'cg19455368', 'cg19595170', 'cg19632760', 'cg19767249', 'cg19831575', 'cg19835478', 'cg19904653', 'cg20011974', 'cg20023578', 'cg20051177', 'cg20139214', 'cg20263942', 'cg20284673', 'cg20287640', 'cg20346096', 'cg20496643', 'cg20525378', 'cg20585500', 'cg20627916', 'cg20728496', 'cg20775254', 'cg20789691', 'cg20795863', 'cg20881910', 'cg20932765', 'cg20969242', 'cg21092324', 'cg21165219', 'cg21206959', 'cg21289015', 'cg21298523', 'cg21504624', 'cg21611708', 'cg21618439', 'cg21642649', 'cg21667943', 'cg21678388', 'cg21712678', 'cg21754343', 'cg21820677', 'cg21922841', 'cg21939482', 'cg21968169', 'cg22021786', 'cg22051763', 'cg22283058', 'cg22295573', 'cg22341104', 'cg22377237', 'cg22464186', 'cg22680204', 'cg22800631', 'cg22814929', 'cg22825487', 'cg22926560', 'cg22995176', 'cg23018448', 'cg23032316', 'cg23036025', 'cg23054676', 'cg23074747', 'cg23114594', 'cg23226134', 'cg23240961', 'cg23274244', 'cg23282674', 'cg23306832', 'cg23337754', 'cg23408913', 'cg23520347', 'cg23632840', 'cg23698058', 'cg23735442', 'cg23792364', 'cg23858360', 'cg23896056', 'cg23957915', 'cg24034289', 'cg24084891', 'cg24088229', 'cg24101359', 'cg24117442', 'cg24176037', 'cg24341129', 'cg24400943', 'cg24445405', 'cg24481633', 'cg24497877', 'cg24532669', 'cg24558204', 'cg24687051', 'cg24735937', 'cg24792272', 'cg24832140', 'cg24903376', 'cg25007680', 'cg25017250', 'cg25098644', 'cg25101056', 'cg25149927', 'cg25172835', 'cg25219134', 'cg25302370', 'cg25418831', 'cg25425078', 'cg25527547', 'cg25598083', 'cg25762395', 'cg25788012', 'cg25802871', 'cg25859012', 'cg25922239', 'cg25956985', 'cg25969212', 'cg25985103', 'cg25990230', 'cg25999267', 'cg26020513', 'cg26045205', 'cg26069745', 'cg26097271', 'cg26199493', 'cg26266098', 'cg26357453', 'cg26530200', 'cg26637901', 'cg26647453', 'cg26665419', 'cg26729026', 'cg26764555', 'cg26767761', 'cg26790132', 'cg26809210', 'cg26820922', 'cg26850754', 'cg27015174', 'cg27091343', 'cg27363310', 'cg27376817', 'cg27398547', 'cg27519373', 'cg27519424', 'cg27566805', 'cg27655905']
missing_horvath_cpgs = ['cg02654291', 'cg02972551', 'cg09785172', 'cg09869858', 'cg13682722', 'cg14329157', 'cg16494477', 'cg17408647', 'cg19167673', 'cg19273182', 'cg19945840', 'cg20795863', 'cg27319898', 'cg01511567', 'cg04431054', 'cg05590257', 'cg06117855', 'cg11388238', 'cg14423778', 'cg19046959', 'cg19569684', 'cg24471894', 'cg27016307']
missing_hannum_cpgs = ['ch.13.39564907R', 'cg14361627', 'cg21139312', 'cg09651136', 'ch.2.30415474F', 'cg18473521', 'cg07927379', 'cg24079702', 'cg25428494']
missing_horvath_sb_cpgs = ['cg07303143', 'cg14614643', 'cg11620135', 'cg01892695', 'cg13767001', 'cg26311454', 'cg06737494', 'cg02901139']
missing_grim_cpgs = ['cg20569940', 'cg26156167', 'cg20800892', 'cg03274876', 'cg17533522', 'cg14868212', 'cg13309828', 'cg03415429', 'cg15947697', 'cg11867651', 'cg07205627', 'cg12211040', 'cg01624571', 'cg08072101', 'cg03400403', 'cg21736089', 'cg03706056'] + ['cg06901711', 'cg24110177', 'cg10177080', 'cg15845365', 'cg02081065', 'cg03807873', 'cg07025011', 'cg02716826', 'cg13898384', 'cg09845806'] + ['cg21789941', 'cg13371627', 'cg00684178', 'cg03900798', 'cg17543884', 'cg15822010', 'cg24493971', 'cg16649728', 'cg05021075', 'cg01641177', 'cg07660627', 'cg23549061'] + ['cg16511983', 'cg01491219', 'cg05515143', 'cg00706683', 'cg19393314', 'cg21770393', 'cg01206872', 'cg14924781', 'cg01704252', 'cg07338119', 'cg19012696', 'cg12465010', 'cg06809342', 'cg04265051', 'cg12051762', 'cg17116694', 'cg25939203', 'cg23047825', 'cg03294557', 'cg27087885', 'cg20448001', 'cg15480287'] + ['cg25149516', 'cg11782409', 'cg07274490', 'cg07929447'] + ['cg04659537', 'cg12791136', 'cg12918464', 'cg00398048', 'cg09578605', 'cg21584251', 'cg04599158', 'cg02627240', 'cg01026009', 'cg03276920', 'cg22355889', 'cg26527903', 'cg13611456', 'cg27294156', 'cg06133392', 'cg23573129', 'cg09219182', 'cg18476633', 'cg26341773', 'cg25034591', 'cg03102848', 'cg18154457', 'cg11924796', 'cg08871010', 'cg06780601', 'cg27395754', 'cg02587153', 'cg15192905', 'cg08352336', 'cg01997410', 'cg16730484', 'cg22686892', 'cg14073590', 'cg07989867', 'cg21080294', 'cg14680768', 'cg10142252', 'cg18717447', 'cg03960747', 'cg16832801'] + ['cg24145109', 'cg08926056', 'cg05971102', 'cg12977946', 'cg06841024', 'cg10407935', 'cg05890887', 'cg12716346', 'cg21227060', 'cg11074353', 'cg07665217', 'cg02505126', 'cg21462914', 'cg19328485', 'cg02580722', 'cg10530344', 'cg21658515', 'cg09351263', 'cg22510362', 'cg00892703', 'cg08667899', 'cg23083672', 'cg02758183', 'cg20308511', 'cg21412053', 'cg26071410', 'cg09625066', 'cg11235848', 'cg02311013', 'cg21664443', 'cg05127178', 'cg19110434', 'cg25697726', 'cg03204600', 'cg11049305', 'cg13501581', 'cg01564693', 'cg10509982', 'cg20557017', 'cg05373692', 'cg05470074', 'cg08262933', 'cg15062055', 'cg12889449', 'cg23955970', 'cg05530348', 'cg11595135', 'cg06298190', 'cg20686207', 'cg14686949', 'cg18003791', 'cg05295197', 'cg23106779'] + ['cg04974804', 'cg12386614', 'cg13678787', 'cg03913456', 'cg24506130', 'cg05308744', 'cg06813250', 'cg07951602', 'cg26816491', 'cg06176987', 'cg22744079', 'cg12599971', 'cg05466385', 'cg26905845', 'cg03730314', 'cg19196326', 'cg17220237', 'cg25977769', 'cg07592681', 'cg14457452', 'cg22660341', 'cg01508796', 'cg17044529', 'cg01802397', 'cg01308343', 'cg15471661', 'cg16558846', 'cg25291250', 'cg02193806', 'cg05815247', 'cg18593194', 'cg22871721', 'cg17215151', 'cg22249566', 'cg17511128', 'cg02495445', 'cg16088894', 'cg13573587', 'cg03788610', 'cg03990195', 'cg08147391', 'cg05548952']
missing_cpgs = list(set(missing_altum_cpgs + missing_hannum_cpgs + missing_horvath_cpgs + missing_horvath_sb_cpgs + missing_grim_cpgs))

organs = ['Blood', 'Breast', 'Kidney', 'Lung', 'Muscle', 'Ovary', 'Prostate', 'Testis', 'Colon', 'All Data']

# Function to read lines from a file and return them as a list
def read_lines_from_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        lines = [line.strip() for line in lines]  # Remove newlines and leading/trailing whitespaces
        return lines

# Directory path where your files are located
directory_path = 'cpg lists'

# Get a list of all files in the directory
file_paths = glob.glob(os.path.join(directory_path, '*.csv'))  # Change '*.txt' to match your file extension if needed

# List to store all the lines from all files as lists
ensemble_cpgs = []

# Loop through each file and read its content line by line
for file_path in file_paths:
    lines = read_lines_from_file(file_path)
    ensemble_cpgs.extend(lines)

ensemble_cpgs = list(set(ensemble_cpgs))

result = pd.DataFrame()
result['cpgs'] = ensemble_cpgs
all_ensemble_preds = np.empty(shape=(0,))
all_cpg_vals = np.empty(shape=(0,21618))
for organ in organs:
    if organ == 'All Data':
        # Calculate the Pearson correlation coefficient between X and Y
        correlation_matrix = np.corrcoef(all_cpg_vals, all_ensemble_preds.reshape(-1, 1).astype(float), rowvar=False)

        # Extract the correlation coefficients between X features and Y features
        correlations_X_Y = correlation_matrix[:-1, -1]

        # Now, 'correlations_X_Y' contains the Pearson correlation coefficients between the features of X and the features of Y.
        result[organ + ' pearson coeff'] = list(correlations_X_Y)
        continue
    df = pd.read_table('input/GTEx/GTEx_' + organ + '.meth.csv')
    real_age, real_gender = get_true_age_labels('input/GTEx/GTEx_' + organ + '.anno.csv')
    missing_rows = [[cpg] + [0.5] * (len(df.columns) - 1) for cpg in missing_cpgs]
    df = pd.concat([df, pd.DataFrame(missing_rows, columns=df.columns)], ignore_index=True)
    df = pd.concat([df, pd.DataFrame([['Age'] + list(real_age)], columns=df.columns)], ignore_index=True)
    df = pd.concat([df, pd.DataFrame([['Female'] + list(real_gender)], columns=df.columns)], ignore_index=True)
    Y = get_ensemble_age(df)
    all_ensemble_preds = np.append(all_ensemble_preds, Y, axis=0)
    Y = Y.reshape(-1, 1).astype(float)
    df.set_index('cgID', inplace=True)
    X = np.array(df.loc[ensemble_cpgs]).T.astype(float)
    all_cpg_vals = np.concatenate((all_cpg_vals, X), axis=0)
    # Calculate the Pearson correlation coefficient between X and Y
    correlation_matrix = np.corrcoef(X, Y, rowvar=False)

    # Extract the correlation coefficients between X features and Y features
    correlations_X_Y = correlation_matrix[:-1, -1]

    # Now, 'correlations_X_Y' contains the Pearson correlation coefficients between the features of X and the features of Y.
    result[organ + ' pearson coeff'] = list(correlations_X_Y)
    print(all_cpg_vals.shape, all_ensemble_preds.shape)
    print(result)

result.to_csv('PearsonCoeffEnsemble.csv')