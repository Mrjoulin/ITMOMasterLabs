import React, { useState, useRef, useEffect } from 'react';
import { AppData, Task, Priority, Area, SearchResult, Note } from '../types';
import { api } from '../services/api';

interface TasksPageProps {
  data: AppData | null;
  onDataUpdate: () => void;
  onOpenTask: (task: Task) => void;
  onNewTask: (areaId?: number | null) => void;
  onOpenNote: (note: Note) => void;
}

const TasksPage: React.FC<TasksPageProps> = ({ data, onDataUpdate, onOpenTask, onNewTask, onOpenNote }) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [isSearchLoading, setIsSearchLoading] = useState(false);
  const [noSearchResults, setNoSearchResults] = useState(false);
  const searchRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handler = setTimeout(async () => {
      if (searchQuery.length > 1) {
        setIsSearchLoading(true);
        try {
          const results = await api.search(searchQuery, 'task');
          setSearchResults(results);
          setNoSearchResults(results.length === 0);
        } catch (error) {
          console.error("Search failed:", error);
          setSearchResults([]);
          setNoSearchResults(true);
        } finally {
          setIsSearchLoading(false);
        }
      } else {
        setSearchResults([]);
        setNoSearchResults(false);
      }
    }, 300); // 300ms debounce

    return () => {
      clearTimeout(handler);
    };
  }, [searchQuery]);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
        if (searchRef.current && !searchRef.current.contains(event.target as Node)) {
            setSearchResults([]);
            setNoSearchResults(false);
            setSearchQuery(''); // Clear search query on click outside
        }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const handleOpenResult = (result: SearchResult) => {
    if (!data) return;
    if (result.type === 'task') {
      const task = data.tasks.find(t => t.id === result.id);
      if (task) {
        onOpenTask(task);
      }
    }
    setSearchResults([]);
    setNoSearchResults(false);
    setSearchQuery('');
  };


  if (!data) return <div>Loading...</div>;

  const handleToggle = async (taskId: string, completed: boolean, e: React.MouseEvent) => {
    e.stopPropagation();
    try {
        await api.toggleTaskCompletion(taskId, completed);
        onDataUpdate();
    } catch (err) {
        console.error(err);
    }
  };

  const pendingTasks = data.tasks.filter(t => !t.completed);
  const completedTasks = data.tasks.filter(t => t.completed);

  return (
    <div className="px-6 py-8 h-full overflow-y-auto">
      {/* Header */}
      <header className="flex flex-col md:flex-row md:items-center justify-between mb-8 gap-4">
        <div>
          <h2 className="text-3xl font-bold text-slate-900 dark:text-white">My Tasks</h2>
          <p className="text-slate-500 dark:text-slate-400 mt-1 flex items-center gap-1 text-sm">
            <span className="material-icons text-base">calendar_today</span>
            <span>Today, {new Date().toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}</span>
          </p>
        </div>
        <div className="flex items-center gap-3 relative" ref={searchRef}>
          <div className="relative">
            <span className="absolute inset-y-0 left-0 flex items-center pl-3 text-slate-400">
              <span className="material-icons text-[20px]">search</span>
            </span>
            <input
              className="w-48 pl-10 pr-4 py-2.5 rounded-lg border border-slate-200 dark:border-slate-700 bg-white dark:bg-neutral-surface-dark text-sm focus:outline-none focus:ring-2 focus:ring-primary/50 transition-shadow shadow-sm outline-none"
              placeholder="Search tasks..."
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              onFocus={() => searchQuery.length > 1 && setSearchResults(searchResults)}
            />
            {isSearchLoading && (
                <span className="absolute inset-y-0 right-0 flex items-center pr-3">
                <span className="animate-spin rounded-full h-4 w-4 border-t-2 border-b-2 border-primary"></span>
                </span>
            )}
          </div>
          {searchResults.length > 0 && searchQuery.length > 1 && (
            <div className="absolute left-0 right-0 top-full mt-2 bg-white dark:bg-slate-800 rounded-lg shadow-xl border border-slate-200 dark:border-slate-700 z-30 max-h-80 overflow-y-auto animate-fade-in-down">
                {searchResults.map((result) => (
                <button
                    key={result.type + result.id}
                    onClick={() => handleOpenResult(result)}
                    className="w-full text-left p-3 hover:bg-slate-50 dark:hover:bg-slate-700 flex flex-col transition-colors border-b border-slate-100 dark:border-slate-700 last:border-b-0"
                >
                    <span className="text-xs text-slate-500 dark:text-slate-400 uppercase font-semibold">{result.type}</span>
                    <h4 className="font-semibold text-slate-900 dark:text-white truncate">{result.title}</h4>
                    {result.type === 'task' && result.description && (
                    <p className="text-sm text-slate-500 dark:text-slate-400 line-clamp-1">{result.description}</p>
                    )}
                </button>
                ))}
            </div>
            )}
            {searchQuery.length > 1 && !isSearchLoading && noSearchResults && (
                <div className="absolute left-0 right-0 top-full mt-2 bg-white dark:bg-slate-800 rounded-lg shadow-xl border border-slate-200 dark:border-slate-700 z-30 p-3 text-center text-sm text-slate-500 animate-fade-in-down">
                    No results found.
                </div>
            )}
          <button onClick={() => onNewTask(null)} className="bg-primary hover:bg-blue-600 text-white px-5 py-2.5 rounded-lg font-medium shadow-lg shadow-primary/30 flex items-center gap-2 transition-all active:scale-95">
            <span className="material-icons text-sm">add</span>
            <span>Add Task</span>
          </button>
        </div>
      </header>

      {/* Grouped Tasks */}
      {data.areas.map(area => {
        const areaTasks = pendingTasks.filter(t => t.area_id === area.id);

        const priorityOrder: Record<Priority, number> = {
            [Priority.High]: 3,
            [Priority.Medium]: 2,
            [Priority.Low]: 1,
        };

        const sortedAreaTasks = [...areaTasks].sort((a, b) => {
            // Sort by priority (High to Low)
            const priorityA = a.priority ? priorityOrder[a.priority] : 0;
            const priorityB = b.priority ? priorityOrder[b.priority] : 0;
            if (priorityA !== priorityB) {
                return priorityB - priorityA;
            }

            // For tasks with same priority, sort by due_date (lowest to highest)
            if (a.due_date && b.due_date) {
                return new Date(a.due_date).getTime() - new Date(b.due_date).getTime();
            }
            // Tasks with a due date come before tasks without a due date
            if (a.due_date && !b.due_date) {
                return -1;
            }
            if (!a.due_date && b.due_date) {
                return 1;
            }

            // Fallback to created_at if no priority or due_date
            return new Date(b.created_at).getTime() - new Date(a.created_at).getTime();
        });

        if (areaTasks.length === 0) return null;

        return (
            <section key={area.id} className="mb-10 animate-fade-in-up">
                <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center gap-3">
                        <span className={`w-3 h-3 rounded-full ${area.color} shadow-sm`}></span>
                        <h3 className="text-lg font-bold text-slate-800 dark:text-slate-100">{area.name} Area</h3>
                        <span className="bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400 px-2.5 py-0.5 rounded-md text-xs font-semibold">{areaTasks.length}</span>
                    </div>
                </div>
                <div className="grid gap-3">
                    {sortedAreaTasks.map(task => (
                        <div 
                            key={task.id} 
                            onClick={() => onOpenTask(task)}
                            className="group bg-white dark:bg-neutral-surface-dark rounded-xl p-4 border border-slate-200 dark:border-slate-800 shadow-sm hover:shadow-md hover:border-primary/30 transition-all duration-200 cursor-pointer flex items-start gap-4"
                        >
                            <div className="pt-1" onClick={(e) => handleToggle(task.id, task.completed, e)}>
                                <div className="w-5 h-5 rounded border border-slate-300 text-primary cursor-pointer hover:border-primary flex items-center justify-center"></div>
                            </div>
                            <div className="flex-1">
                                <div className="flex items-start justify-between">
                                    <h4 className="font-semibold text-slate-900 dark:text-white group-hover:text-primary transition-colors">{task.title}</h4>
                                    {task.priority && (
                                        <span className={`text-[10px] font-bold px-2 py-1 rounded uppercase tracking-wide ${
                                            task.priority === Priority.High ? 'bg-red-50 text-red-600 dark:bg-red-900/20 dark:text-red-400' : 
                                            task.priority === Priority.Medium ? 'bg-blue-50 text-blue-600 dark:bg-blue-900/20 dark:text-blue-400' : 
                                            'bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-400'
                                        }`}>{task.priority} Priority</span>
                                    )}
                                </div>
                                <p className="text-slate-500 dark:text-slate-400 text-sm mt-1 line-clamp-1">{task.description}</p>
                                {task.due_date && (
                                    <div className="flex items-center gap-3 mt-3">
                                        <span className="flex items-center text-xs text-slate-400">
                                            <span className="material-icons text-[14px] mr-1">schedule</span>
                                            {new Date(task.due_date).toLocaleDateString()}
                                        </span>
                                    </div>
                                )}
                            </div>
                        </div>
                    ))}
                </div>
            </section>
        );
      })}

      {/* Completed */}
      {completedTasks.length > 0 && (
         <section className="border-t border-slate-200 dark:border-slate-800 pt-8 opacity-60 hover:opacity-100 transition-opacity duration-300">
            <div className="flex items-center gap-2 mb-4 cursor-pointer">
                <span className="material-icons text-slate-400 text-lg">expand_more</span>
                <h3 className="text-sm font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wider">Completed Tasks ({completedTasks.length})</h3>
            </div>
            <div className="grid gap-3">
                {completedTasks.map(task => (
                     <div key={task.id} className="bg-slate-50 dark:bg-slate-900/50 rounded-xl p-4 border border-slate-100 dark:border-slate-800 flex items-start gap-4" onClick={(e) => handleToggle(task.id, task.completed, e)}>
                        <div className="pt-1">
                            <div className="w-5 h-5 rounded border border-slate-300 bg-slate-200 dark:bg-slate-700 flex items-center justify-center">
                                <span className="material-icons text-sm text-slate-500">check</span>
                            </div>
                        </div>
                        <div className="flex-1">
                            <div className="flex items-start justify-between">
                                <h4 className="font-medium text-slate-500 line-through">{task.title}</h4>
                            </div>
                        </div>
                     </div>
                ))}
            </div>
         </section>
      )}
    </div>
  );
};

export default TasksPage;
