from CodeAnalysis import ProjectAnalyzer



if __name__ == '__main__':
    from Inspection.utils.path_manager import DEMO_PATH
    path = DEMO_PATH + 'DemoAIProject'
    project_analyzer = ProjectAnalyzer(path,record_debug_info=True,record_base_dir=DEMO_PATH + 'AnalysiserResult/')
    project_analyzer.run()
    project_analyzer.save_analysis_result()

