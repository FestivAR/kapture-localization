import argparse
import logging
import os
import os.path as path
import sys
import subprocess
from typing import List, Optional

sys.path.append(path.dirname(path.abspath(path.dirname(__file__))))

from pipeline import pipeline_import_paths  # noqa: F401

import kapture_localization.utils.logging
from kapture_localization.utils.symlink import can_use_symlinks, create_kapture_proxy_single_features
from kapture_localization.utils.subprocess import run_python_command
from kapture_localization.colmap.colmap_command import CONFIGS

import kapture_localization.utils.path_to_kapture  # noqa: F401
import kapture.utils.logging
from kapture.utils.paths import safe_remove_file

logger = logging.getLogger('localize_image')


def localize_image_pipeline(lfeat_ext_path: str,
                            gfeat_ext_path: str,
                            kapture_map_path: str,
                            query_image_path: str,
                            keypoints_path: str,
                            descriptors_path: str,
                            global_features_path: str,
                            matches_path: str,
                            matches_gv_path: str,
                            colmap_map_path: str,
                            localization_output_path: str,
                            colmap_binary: str,
                            python_binary: Optional[str],
                            topk: int,
                            config: int,
                            force_overwrite_existing: bool) -> None:
    keypoints_type = None
    descriptors_type = None
    global_features_type = None

    os.makedirs(localization_output_path, exist_ok=True)
    pairsfile_path = path.join(localization_output_path, f'pairs_localization_{topk}.txt')
    map_plus_query_path = path.join(localization_output_path, 'kapture_inputs/map_plus_query')
    colmap_localize_path = path.join(localization_output_path, f'colmap_localized')
    os.makedirs(colmap_localize_path, exist_ok=True)
    kapture_localize_import_path = path.join(localization_output_path, f'kapture_localized')
    kapture_localize_recover_path = path.join(localization_output_path, f'kapture_localized_recover')

    if not path.isdir(matches_path):
        os.makedirs(matches_path)
    if not path.isdir(matches_gv_path):
        os.makedirs(matches_gv_path)

    # make kapture data from query image
    # kapture_import_image_path = path.join(pipeline_import_paths.HERE_PATH,
    #                                       '../../kapture/tools/kapture_import_image_folder.py')
    kapture_query_path = path.join(kapture_map_path, '../query')
    import_image_args = ['-i', query_image_path,
                         '-o', kapture_query_path]
    if force_overwrite_existing:
        import_image_args.append('-f')
    run_python_command('kapture_import_image_folder.py', import_image_args)

    # extract local features. 
    # Only for r2d2 with pre defined settings. must be same with mapping kapture extracter settings.
    # lfeat_extracter_path = path.join(lfeat_ext_path, 'extract_kapture_modif.py')
    lfeat_extracter_path = path.join(lfeat_ext_path, 'extract_kapture_modif.py')
    lfeat_ext_args = ['python', lfeat_extracter_path,
                      '--model', path.join(lfeat_ext_path, 'models/faster2d2_WASF_N16.pt'),
                      '--kapture-root', kapture_query_path,
                      '--top-k', '5000',
                      '--gpu', '0']
    # extract global features.
    # Only for deep-image-retrival with pre defined settings. must be same with kapture extracter settings.
    gfeat_extracter_path = path.abspath(gfeat_ext_path)
    gfeat_ext_args = ['python', 
                      '-m', 'dirtorch.extract_kapture_modif',
                      '--kapture-root', path.abspath(kapture_query_path), 
                      '--checkpoint', path.join(gfeat_extracter_path, 'dirtorch/data/Resnet101-AP-GeM-LM18.pt'),
                      '--gpu', '0']

    use_shell = sys.platform.startswith("win")
    python_process = [subprocess.Popen(lfeat_ext_args, shell=use_shell),
                      subprocess.Popen(gfeat_ext_args, shell=use_shell, cwd=gfeat_extracter_path)]
    for proc in python_process:
        proc.wait()
        if proc.returncode != 0:
            raise ValueError('\nSubprocess Error (Return code:' f' {proc.returncode} )')

    # build proxy kapture map in output folder
    proxy_kapture_map_path = path.join(localization_output_path, 'kapture_inputs/proxy_mapping')
    create_kapture_proxy_single_features(proxy_kapture_map_path,
                                         kapture_map_path,
                                         keypoints_path,
                                         descriptors_path,
                                         global_features_path,
                                         matches_path,
                                         keypoints_type,
                                         descriptors_type,
                                         global_features_type,
                                         force_overwrite_existing)

    # build proxy kapture query in output folder
    proxy_kapture_query_path = path.join(localization_output_path, 'kapture_inputs/proxy_query')
    create_kapture_proxy_single_features(proxy_kapture_query_path,
                                         kapture_query_path,
                                         keypoints_path,
                                         descriptors_path,
                                         global_features_path,
                                         matches_path,
                                         keypoints_type,
                                         descriptors_type,
                                         global_features_type,
                                         force_overwrite_existing)

    # kapture_compute_image_pairs.py
    local_image_pairs_path = path.join(pipeline_import_paths.HERE_PATH, '../tools/kapture_compute_image_pairs.py')
    if os.path.isfile(pairsfile_path):
        safe_remove_file(pairsfile_path, force_overwrite_existing)
    compute_image_pairs_args = ['-v', str(logger.level),
                                '--mapping', proxy_kapture_map_path,
                                '--query', proxy_kapture_query_path,
                                '--topk', str(topk),
                                '-o', pairsfile_path]
    run_python_command(local_image_pairs_path, compute_image_pairs_args, python_binary)

    # kapture_merge.py
    local_merge_path = path.join(pipeline_import_paths.HERE_PATH, '../../kapture/tools/kapture_merge.py')
    merge_args = ['-v', str(logger.level),
                  '-i', proxy_kapture_map_path, proxy_kapture_query_path,
                  '-o', map_plus_query_path,
                  '-s', 'keypoints', 'descriptors', 'global_features', 'matches',
                  '--image_transfer', 'link_absolute']
    if force_overwrite_existing:
        merge_args.append('-f')
    run_python_command(local_merge_path, merge_args, python_binary)

    # build proxy kapture map+query in output folder
    proxy_kapture_map_plus_query_path = path.join(localization_output_path, 'kapture_inputs/proxy_map_plus_query')
    create_kapture_proxy_single_features(proxy_kapture_map_plus_query_path,
                                         map_plus_query_path,
                                         keypoints_path,
                                         descriptors_path,
                                         global_features_path,
                                         matches_path,
                                         keypoints_type,
                                         descriptors_type,
                                         global_features_type,
                                         force_overwrite_existing)

    # kapture_compute_matches.py
    local_compute_matches_path = path.join(pipeline_import_paths.HERE_PATH, '../tools/kapture_compute_matches.py')
    compute_matches_args = ['-v', str(logger.level),
                            '-i', proxy_kapture_map_plus_query_path,
                            '--pairsfile-path', pairsfile_path]
    run_python_command(local_compute_matches_path, compute_matches_args, python_binary)

    # build proxy gv kapture in output folder
    proxy_kapture_map_plus_query_gv_path = path.join(localization_output_path, 'kapture_inputs/proxy_map_plus_query_gv')
    create_kapture_proxy_single_features(proxy_kapture_map_plus_query_gv_path,
                                         map_plus_query_path,
                                         keypoints_path,
                                         descriptors_path,
                                         global_features_path,
                                         matches_gv_path,
                                         keypoints_type,
                                         descriptors_type,
                                         global_features_type,
                                         force_overwrite_existing)

    # kapture_run_colmap_gv.py
    local_run_colmap_gv_path = path.join(pipeline_import_paths.HERE_PATH, '../tools/kapture_run_colmap_gv.py')
    run_colmap_gv_args = ['-v', str(logger.level),
                          '-i', proxy_kapture_map_plus_query_path,
                          '-o', proxy_kapture_map_plus_query_gv_path,
                          '--pairsfile-path', pairsfile_path,
                          '-colmap', colmap_binary]
    if force_overwrite_existing:
        run_colmap_gv_args.append('-f')
    run_python_command(local_run_colmap_gv_path, run_colmap_gv_args, python_binary)

    # kapture_colmap_localize.py
    local_localize_path = path.join(pipeline_import_paths.HERE_PATH, '../tools/kapture_colmap_localize.py')
    localize_args = ['-v', str(logger.level),
                     '-i', proxy_kapture_map_plus_query_gv_path,
                     '-o', colmap_localize_path,
                     '-colmap', colmap_binary,
                     '--pairs-file-path', pairsfile_path,
                     '-db', path.join(colmap_map_path, 'colmap.db'),
                     '-txt', path.join(colmap_map_path, 'reconstruction')]
    if force_overwrite_existing:
        localize_args.append('-f')
    localize_args += CONFIGS[config]
    run_python_command(local_localize_path, localize_args, python_binary)

    # kapture_import_colmap.py
    local_import_colmap_path = path.join(pipeline_import_paths.HERE_PATH,
                                         '../../kapture/tools/kapture_import_colmap.py')
    import_colmap_args = ['-v', str(logger.level),
                          '-db', path.join(colmap_localize_path, 'colmap.db'),
                          '-txt', path.join(colmap_localize_path, 'reconstruction'),
                          '-o', kapture_localize_import_path,
                          '--skip_reconstruction']
    if force_overwrite_existing:
        import_colmap_args.append('-f')
    run_python_command(local_import_colmap_path, import_colmap_args, python_binary)

    local_recover_path = path.join(pipeline_import_paths.HERE_PATH,
                                   '../tools/kapture_recover_timestamps_and_ids.py')
    recover_args = ['-v', str(logger.level),
                    '-i', kapture_localize_import_path,
                    '--ref', kapture_query_path,
                    '-o', kapture_localize_recover_path,
                    '--image_transfer', 'skip']
    if force_overwrite_existing:
        recover_args.append('-f')
    run_python_command(local_recover_path, recover_args, python_binary)

    # send localization result
    # for now, only 1 last img's traj is available
    try:
        result_file = path.join(kapture_localize_recover_path, 'sensors/trajectories.txt')
        with open(result_file) as file:
            results = file.readlines()
            results = [line.rstrip() for line in results if line != '\n']
            for i in range(0, len(results)):
                if results[i][0] == '#':
                    continue
                else:
                    print(results[i].lstrip())
            return results

    except OSError:
        logger.info('No result trajectories found!')
        print('No result trajectories found!\n')
        return ['0']


def localize_image_pipeline_command_line():
    parser = argparse.ArgumentParser(description='localize a single image on a colmap map')
    parser_verbosity = parser.add_mutually_exclusive_group()
    parser_verbosity.add_argument('-v', '--verbose', nargs='?', default=logging.WARNING, const=logging.INFO,
                                  action=kapture.utils.logging.VerbosityParser,
                                  help='verbosity level (debug, info, warning, critical, ... or int value) [warning]')
    parser_verbosity.add_argument('-q', '--silent', '--quiet', action='store_const',
                                  dest='verbose', const=logging.CRITICAL)
    parser.add_argument('-f', '-y', '--force', action='store_true', default=True,
                        help='silently delete pairfile and localization results if already exists.')
    parser.add_argument('-lfext', '--lfeat-ext-path', required=True,
                        help='path to the local feature extracter') # path to r2d2
    parser.add_argument('-gfext', '--gfeat-ext-path', required=True,
                        help='path to the global feature extractor') # path to deep-image-retrival
    parser.add_argument('-i', '--kapture-map', required=True,
                        help='path to the kapture map directory')
    parser.add_argument('--query-img', required=True,
                        help='input path to query image data root directory')
    parser.add_argument('-kpt', '--keypoints-path', required=True,
                        help='input path to the orphan keypoints folder')
    parser.add_argument('-desc', '--descriptors-path', required=True,
                        help='input path to the orphan descriptors folder')
    parser_pairing = parser.add_mutually_exclusive_group(required=True)
    parser_pairing.add_argument('-gfeat', '--global-features-path', default=None,
                                help='input path to the orphan global features folder')
    parser.add_argument('-matches', '--matches-path', required=True,
                        help='input path to the orphan matches (no geometric verification) folder')
    parser.add_argument('-matches-gv', '--matches-gv-path', required=True,
                        help='input path to the orphan matches (with geometric verification) folder')
    parser.add_argument('--colmap-map', required=True,
                        help='path to the colmap map directory')
    parser.add_argument('-o', '--output', required=True,
                        help='output directory.')
    parser.add_argument('-colmap', '--colmap_binary', required=False,
                        default="colmap",
                        help='full path to colmap binary '
                             '(default is "colmap", i.e. assume the binary'
                             ' is in the user PATH).')
    parser_python_bin = parser.add_mutually_exclusive_group()
    parser_python_bin.add_argument('-python', '--python_binary', required=False,
                                   default=None,
                                   help='full path to python binary '
                                        '(default is "None", i.e. assume the os'
                                        ' can infer the python binary from the files itself, shebang or extension).')
    parser_python_bin.add_argument('--auto-python-binary', action='store_true', default=False,
                                   help='use sys.executable as python binary.')
    default_topk = 20
    parser.add_argument('--topk',
                        default=default_topk,
                        type=int,
                        help='the max number of top retained images when computing image pairs from global features')
    parser.add_argument('--config', default=1, type=int,
                        choices=list(range(len(CONFIGS))), help='what config to use for image registrator')
    args = parser.parse_args()

    logger.setLevel(args.verbose)
    if args.verbose <= logging.INFO:
        # also let kapture express its logs
        kapture.utils.logging.getLogger().setLevel(args.verbose)
        kapture_localization.utils.logging.getLogger().setLevel(args.verbose)

    args_dict = vars(args)
    logger.debug('localize_image.py \\\n' + '  \\\n'.join(
        '--{:20} {:100}'.format(k, str(v)) for k, v in args_dict.items()))
    if can_use_symlinks():
        python_binary = args.python_binary
        if args.auto_python_binary:
            python_binary = sys.executable
            logger.debug(f'python_binary set to {python_binary}')
        localize_image_pipeline(args.lfeat_ext_path,
                                args.gfeat_ext_path,
                                args.kapture_map,
                                args.query_img,
                                args.keypoints_path,
                                args.descriptors_path,
                                args.global_features_path,
                                args.matches_path,
                                args.matches_gv_path,
                                args.colmap_map,
                                args.output,
                                args.colmap_binary,
                                python_binary,
                                args.topk,
                                args.config,
                                args.force)
    else:
        raise EnvironmentError('Please restart this command as admin, it is required for os.symlink'
                               'see https://docs.python.org/3.6/library/os.html#os.symlink')


if __name__ == '__main__':
    localize_image_pipeline_command_line()
